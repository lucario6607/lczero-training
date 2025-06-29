import os
import time
import numpy as np
import tensorflow as tf
import traceback
from chunkparser import create_dataset
import gzip
import re # Import the regular expression module

# --- Import new dependencies for Sparsity and add a check ---
try:
    import tensorflow_model_optimization as tfmot
    sparsity_available = True
except ImportError:
    print("Warning: tensorflow_model_optimization not found. Sparsity will be disabled.")
    sparsity_available = False


# --- Import the CORRECT conversion tools ---
try:
    from net import Net, pb
except ImportError:
    print("Warning: net.py or its dependencies not found. .pb.gz export will be disabled.")
    Net = None
    pb = None


# --- Keras Layers and Schedules ---

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps, name=None):
        super().__init__()
        self.initial_learning_rate = float(initial_learning_rate)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.name = name
    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            step = tf.cast(step, tf.float32)
            def warmup_fn():
                return self.initial_learning_rate * (step / self.warmup_steps)
            def decay_fn():
                decay_steps = tf.maximum(1.0, self.total_steps - self.warmup_steps)
                cosine_decay = 0.5 * (1 + tf.cos(np.pi * (step - self.warmup_steps) / decay_steps))
                return self.initial_learning_rate * cosine_decay
            return tf.cond(step < self.warmup_steps, warmup_fn, decay_fn)
    def get_config(self):
        return {"initial_learning_rate": self.initial_learning_rate, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps, "name": self.name}

class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(input_shape[-1],), initializer="ones", trainable=True)
    def call(self, x):
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        inv_rms = tf.math.rsqrt(tf.maximum(variance, self.eps))
        return x * inv_rms * self.gamma

class Mamba2Block(tf.keras.layers.Layer):
    def __init__(self, d_model, d_state, d_conv, expand, dt_rank, **kwargs):
        super().__init__(**kwargs)
        self.d_i, self.d_s, self.dt_r, self.d_c, self.d_m = int(expand * d_model), min(d_state, 16), dt_rank, d_conv, d_model

    def build(self, input_shape):
        he_init = tf.keras.initializers.HeNormal(seed=42)
        self.norm = RMSNorm(name="norm")
        self.in_proj = tf.keras.layers.Dense(use_bias=False, name="in_proj", kernel_initializer=he_init, units=self.d_i * 2)
        self.conv1d = tf.keras.layers.Conv1D(filters=self.d_i, kernel_size=self.d_c, padding="causal", name="conv1d", activation=None, kernel_initializer=he_init)
        self.x_proj = tf.keras.layers.Dense(use_bias=False, name="x_proj", kernel_initializer=he_init, units=self.dt_r + self.d_s * 2)
        self.dt_proj = tf.keras.layers.Dense(name="dt_proj", kernel_initializer=he_init, bias_initializer=tf.constant_initializer(-2.0), units=self.d_i)
        self.out_proj = tf.keras.layers.Dense(use_bias=False, name="out_proj", kernel_initializer=he_init, units=self.d_m)
        self.D = self.add_weight(name="D", shape=(self.d_i,), initializer=tf.constant_initializer(0.01), trainable=True)
        a_log_init = np.clip(-np.log(np.linspace(0.5, 2.0, self.d_s)), -5.0, 0.0)
        self.A_log = self.add_weight(name="A_log", shape=(self.d_i, self.d_s), initializer=tf.constant_initializer(np.tile(a_log_init, (self.d_i, 1))), trainable=True)
    
    def ssm_manual_loop(self, x, dt, B, C):
        seq_len = tf.shape(x)[1]
        outputs_ta = tf.TensorArray(dtype=self.compute_dtype, size=seq_len)
        h = tf.zeros((tf.shape(x)[0], self.d_i, self.d_s), dtype=self.compute_dtype)
        A = -tf.exp(self.A_log); delta = tf.nn.softplus(self.dt_proj(dt))
        for t in tf.range(seq_len):
            delta_t, x_t, B_t, C_t = delta[:, t], x[:, t], B[:, t], C[:, t]
            delta_B = tf.expand_dims(delta_t, -1) * tf.expand_dims(B_t, 1); delta_A = tf.exp(tf.expand_dims(delta_t, -1) * A)
            delta_B_u = delta_B * tf.expand_dims(x_t, -1); h = delta_A * h + delta_B_u
            y_t = tf.reduce_sum(h * tf.expand_dims(C_t, 1), axis=-1); outputs_ta = outputs_ta.write(t, y_t)
        return tf.transpose(outputs_ta.stack(), perm=[1, 0, 2])

    def call(self, x, training=None):
        residual = x; x = self.norm(x)
        x_proj, z = tf.split(self.in_proj(x), 2, axis=-1); x_conv = tf.nn.swish(self.conv1d(x_proj))
        dt_in, B_in, C_in = tf.split(self.x_proj(x_conv), [self.dt_r, self.d_s, self.d_s], axis=-1)
        y = self.ssm_manual_loop(x_conv, dt_in, B_in, C_in); y = y + x_conv * self.D; y = y * tf.nn.silu(z)
        return residual + self.out_proj(y)

class CheckpointWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs): 
        super().__init__(**kwargs)
        self.layer = layer
    def call(self, inputs, training=None, **kwargs):
        def recompute_fn(x):
            return self.layer(x, training=training, **kwargs)
        return tf.recompute_grad(recompute_fn)(inputs)

# --- Main Training Process Class ---
class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg; self.mcfg = cfg["model"]; self.tcfg = cfg["training"]
        self.root_dir = os.path.join(self.tcfg["path"], self.cfg["name"])
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        self.use_sparsity = False
        self.pruning_params = None

    def construct_net(self):
        mcfg = self.mcfg
        original_policy = tf.keras.mixed_precision.global_policy()
        model_construction_policy = 'float32' if self.use_sparsity else original_policy.name
        
        if self.use_sparsity:
            print(f"Temporarily setting global policy to '{model_construction_policy}' for model construction.")
            tf.keras.mixed_precision.set_global_policy(model_construction_policy)
        
        model = None
        try:
            def prunable_layer_fn(layer_class, **kwargs):
                layer_to_wrap = layer_class(**kwargs)
                if self.pruning_params and isinstance(layer_to_wrap, tf.keras.layers.Dense):
                    try:
                        return tfmot.sparsity.keras.prune_low_magnitude(
                            layer_to_wrap, **self.pruning_params)
                    except Exception as e:
                        print(f"Warning: Could not apply pruning to {layer_class.__name__}: {e}")
                return layer_to_wrap

            inp = tf.keras.Input(shape=(112, 8, 8), name="us_input", dtype=tf.float16)
            
            if self.use_sparsity:
                flow = tf.cast(inp, tf.float32)
            else:
                flow = inp
                
            flow = tf.keras.layers.Permute((2, 3, 1))(flow)
            flow = tf.keras.layers.Reshape((64, 112))(flow)
            
            embedding_size = mcfg['embedding_size']
            input_dense = tf.keras.layers.Dense(name="input/dense", activation='relu', kernel_initializer='he_normal', units=embedding_size)
            flow = input_dense(flow)
            
            use_checkpointing = mcfg.get('mamba2_use_checkpointing', False)

            for i in range(mcfg['encoder_layers']):
                mamba_block = Mamba2Block(embedding_size, mcfg['mamba2_d_state'], mcfg['mamba2_d_conv'], 
                                        mcfg.get('mamba2_expand_factor', 2), mcfg['mamba2_dt_rank'],
                                        name=f"encoder_{i}/mamba2")
                
                if use_checkpointing:
                    mamba_block = CheckpointWrapper(mamba_block, name=f"encoder_{i}/checkpoint")
                
                flow = mamba_block(flow, training=False)
            
            pooled_flow = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(flow)
            
            outputs = {}
            policy_head = prunable_layer_fn(tf.keras.layers.Dense, name="policy/dense", kernel_initializer='he_normal', units=1858)
            value_head = prunable_layer_fn(tf.keras.layers.Dense, name="value/dense", kernel_initializer='he_normal', units=3)
            
            outputs['policy'] = policy_head(pooled_flow, training=False)
            outputs['value_winner'] = value_head(pooled_flow, training=False)
            
            model = tf.keras.Model(inputs=inp, outputs=outputs)
            
        except Exception as e:
            print(f"Error during model construction: {e}")
            traceback.print_exc()
            raise
        finally:
            if self.use_sparsity and original_policy.name != model_construction_policy:
                print(f"Restoring global policy to '{original_policy.name}'.")
                tf.keras.mixed_precision.set_global_policy(original_policy)
        
        return model

    def init(self):
        use_dummy_data = self.tcfg.get('use_dummy_data', False)
        if use_dummy_data:
            print("WARNING: Using dummy data for testing!")
            self.train_ds = self.create_dummy_dataset(self.tcfg['batch_size'])
            self.test_ds = self.create_dummy_dataset(self.tcfg['batch_size'])
        else:
            print("Using real data from chunkparser...")
            self.train_ds = create_dataset(self.tcfg['train_dir'], self.tcfg['batch_size'])
            self.test_ds = create_dataset(self.tcfg['test_dir'], self.tcfg['batch_size'], is_training=False)
        self.train_iter, self.test_iter = iter(self.train_ds), iter(self.test_ds)
        
        self.use_sparsity = self.tcfg.get('sparsity', {}).get('enabled', False)
        if self.use_sparsity:
            if not sparsity_available:
                print("ERROR: Sparsity is enabled in config, but 'tensorflow_model_optimization' is not installed. Disabling sparsity.")
                self.use_sparsity = False
            else:
                s_cfg = self.tcfg['sparsity']
                
                print("Applying magnitude-based sparsity to compatible layers.")
                if s_cfg.get('type') == 'block':
                    print("Note: 'block' sparsity type was requested but is known to be unstable. "
                          "Falling back to standard magnitude pruning.")
                
                self.pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.0,
                        final_sparsity=s_cfg.get('target_sparsity', 0.5),
                        begin_step=s_cfg.get('start_step', 1000),
                        end_step=s_cfg.get('end_step', 10000),
                        frequency=s_cfg.get('frequency', 100)
                    )
                }
                
                self.pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

        self.model = self.construct_net()
        print(f"Model created with {self.model.count_params():,} parameters.")

        if self.use_sparsity:
            try:
                print("--- Pruning Summary ---")
                tfmot.sparsity.keras.pruning_summary(self.model)
                print("---------------------")
            except Exception as e:
                print(f"Could not print pruning summary: {e}")

        initial_lr = self.tcfg.get("learning_rate", 1e-4)
        warmup_steps = self.tcfg.get("warmup_steps", 0)
        if warmup_steps > 0:
            print(f"Using learning rate schedule with {warmup_steps} warmup steps.")
            lr_schedule = WarmupCosineDecay(initial_lr, warmup_steps, self.tcfg['total_steps'])
        else:
            print("Using fixed learning rate.")
            lr_schedule = initial_lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0, epsilon=1e-7)
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
        self.global_step = self.optimizer.iterations
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.root_dir, max_to_keep=5, checkpoint_name=self.cfg["name"])
        
        @tf.function
        def train_step_fn(inputs_dict):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs_dict['us'], training=True)
                losses = {}
                policy_loss = tf.reduce_mean(self._policy_loss_fn(inputs_dict['pi'], predictions['policy']))
                value_loss = tf.reduce_mean(self._value_loss_fn(inputs_dict['wdl'], predictions['value_winner']))
                losses['policy'] = policy_loss
                losses['value_winner'] = value_loss
                total_loss = self._lossMix(losses)
                total_loss = tf.cast(total_loss, tf.float32)
                
                weight_decay_val = float(self.tcfg.get("weight_decay", 1e-5))
                if weight_decay_val > 0:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables 
                                      if 'bias' not in v.name and len(v.shape) > 1])
                    total_loss += weight_decay_val * l2_loss
                
                total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, 1e-3)
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            losses['total'] = total_loss
            return losses
        self.train_step = train_step_fn
        
        @tf.function
        def test_step_fn(inputs_dict):
            predictions = self.model(inputs_dict['us'], training=False)
            losses = {}
            losses['policy'] = tf.reduce_mean(self._policy_loss_fn(inputs_dict['pi'], predictions['policy']))
            losses['value_winner'] = tf.reduce_mean(self._value_loss_fn(inputs_dict['wdl'], predictions['value_winner']))
            losses['total'] = self._lossMix(losses)
            return losses
        self.test_step = test_step_fn
    
    def _policy_loss_fn(self, target, pred):
        return tf.keras.losses.categorical_crossentropy(target, pred, from_logits=True)
    
    def _value_loss_fn(self, target, pred):
        return tf.keras.losses.categorical_crossentropy(target, pred, from_logits=True)
    
    def _lossMix(self, losses):
        policy_weight = float(self.tcfg.get('policy_loss_weight', 1.0))
        value_weight = float(self.tcfg.get('value_loss_weight', 1.0))
        return policy_weight * losses['policy'] + value_weight * losses['value_winner']
    
    def export_leela_weights(self, step):
        if Net is None or pb is None: 
            print("Skipping export - net.py not available")
            return
        print(f"Exporting network to .pb.gz format for step {step}...")
        try:
            model_to_export = self.model
            if self.use_sparsity and sparsity_available:
                print("Stripping pruning wrappers for export...")
                model_to_export = tfmot.sparsity.keras.strip_pruning(self.model)

            # FIX: Clean the weight names before passing them to the conversion script.
            # Keras wrappers (for pruning, checkpointing) add prefixes to variable names
            # that the external `net.py` script does not recognize. This routine
            # removes those prefixes.
            tf_weights_dict = {}
            for v in model_to_export.trainable_variables:
                original_name = v.name
                cleaned_name = original_name

                # 1. Remove the pruning wrapper prefix, e.g., "prune_low_magnitude_policy/dense/..."
                if cleaned_name.startswith('prune_low_magnitude_'):
                    cleaned_name = cleaned_name.replace('prune_low_magnitude_', '', 1)

                # 2. Remove the checkpoint wrapper prefix, e.g., "encoder_0/checkpoint/..."
                cleaned_name = re.sub(r'encoder_\d+/checkpoint/', '', cleaned_name)

                tf_weights_dict[cleaned_name] = v.numpy()

            net = Net(net_fmt=pb.NetworkFormat.NETWORK_MAMBA2_WITH_HEADFORMAT)
            net.set_input(pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2)
            net.set_valueformat(pb.NetworkFormat.VALUE_WDL)
            net.populate_from_tf_weights(tf_weights_dict)
            filepath = os.path.join(self.root_dir, f"{self.cfg['name']}-step{step}.pb.gz")
            net.save_proto(filepath)
            print(f"Successfully exported to {filepath}")
        except Exception as e:
            print(f"ERROR during .pb.gz export: {e}")
            traceback.print_exc()
    
    def create_dummy_dataset(self, batch_size):
        def generator():
            while True:
                us = np.random.randn(batch_size, 112, 8, 8).astype(np.float32) * 0.1
                pi = np.random.rand(batch_size, 1858).astype(np.float32)
                pi = pi / np.sum(pi, axis=1, keepdims=True)
                wdl = np.zeros((batch_size, 3), dtype=np.float32)
                for i in range(batch_size): 
                    wdl[i, np.random.randint(0, 3)] = 1.0
                q = np.random.randn(batch_size, 1).astype(np.float32)
                st_q = np.random.randn(batch_size, 1).astype(np.float32)
                yield us, pi, wdl, q, st_q
        return tf.data.Dataset.from_generator(
            generator, 
            output_signature=(
                tf.TensorSpec(shape=(batch_size, 112, 8, 8), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 1858), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32),
            )
        )
    
    def restore(self):
        if self.manager.latest_checkpoint:
            print(f"Restoring from {self.manager.latest_checkpoint}...")
            self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
    
    def evaluate(self):
        print(f"Running evaluation for {self.tcfg['test_steps']} steps...")
        test_losses = {'total': [], 'policy': [], 'value_winner': []}
        for _ in range(self.tcfg['test_steps']):
            try:
                batch_data = next(self.test_iter, None)
                if batch_data is None: 
                    self.test_iter = iter(self.test_ds)
                    batch_data = next(self.test_iter, None)
                if batch_data is None: 
                    break
                us, pi, wdl, q, st_q = batch_data
                input_dict = {'us': us, 'pi': pi, 'wdl': wdl}
                losses = self.test_step(input_dict)
                for k, v in losses.items(): 
                    test_losses[k].append(v.numpy())
            except Exception as e: 
                print(f"Error during evaluation step: {e}")
                continue
        avg_losses = {k: np.mean(v) for k, v in test_losses.items() if v}
        return avg_losses
    
    def debug_data_source(self):
        print(f"=== DEBUGGING DATA SOURCE ===")
        train_dirs = self.tcfg['train_dir']
        if not isinstance(train_dirs, list): 
            train_dirs = [train_dirs]
        print(f"Train Dirs: {train_dirs}")
        print(f"Test Dir: {self.tcfg.get('test_dir', 'Not specified')}")
        print(f"Batch size: {self.tcfg['batch_size']}")
        total_files = 0
        all_dirs_ok = True
        for train_dir in train_dirs:
            print(f"\n--- Checking directory: {train_dir} ---")
            if not os.path.exists(train_dir): 
                print(f"ERROR: Train directory does not exist: {train_dir}")
                all_dirs_ok = False
                continue
            try:
                files = os.listdir(train_dir)
                print(f"Found {len(files)} files in this directory.")
                if not files: 
                    print(f"WARNING: Directory is empty: {train_dir}")
                    continue
                total_files += len(files)
                print(f"  First few files: {files[:5]}")
                for filename in files[:3]:
                    filepath = os.path.join(train_dir, filename)
                    size = os.path.getsize(filepath)
                    print(f"    - {filename}: {size:,} bytes")
                    if size == 0: 
                        print(f"      WARNING: File {filename} is empty!")
            except Exception as e: 
                print(f"ERROR: Could not access directory {train_dir}. Reason: {e}")
                all_dirs_ok = False
                continue
        if not all_dirs_ok: 
            print("\nERROR: One or more directories could not be accessed.")
            return False
        if total_files == 0: 
            print("\nERROR: No training files found in any of the specified directories.")
            return False
        print(f"\nSuccessfully checked all directories. Found a total of {total_files} files.")
        return True
    
    def process_loop(self):
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs("leelalogs", exist_ok=True)
        print(f"Starting training for {self.tcfg['total_steps']} steps...")
        if not self.tcfg.get('use_dummy_data', False) and not self.debug_data_source(): 
            print("Data source debugging failed. Cannot continue training.")
            return
        last_log_time = time.time()
        consecutive_zero_batches = 0
        max_zero_batches = 10
        
        if self.use_sparsity: 
            self.pruning_callback.set_model(self.model)
            self.pruning_callback.on_train_begin()
        
        try:
            sample_batch = next(self.train_iter)
            us, pi, wdl, q, st_q = sample_batch
            print(f"Sample batch shapes - US: {us.shape}, PI: {pi.shape}, WDL: {wdl.shape}")
            if (tf.reduce_sum(tf.abs(us)) < 1e-8 and tf.reduce_sum(tf.abs(pi)) < 1e-8 and 
                tf.reduce_sum(tf.abs(wdl)) < 1e-8): 
                print("ERROR: All sample data is zeros!")
                return
            self.train_iter = iter(self.train_ds)
        except StopIteration: 
            print("ERROR: Data iterator is empty. No data found.")
            return
        except Exception as e: 
            print(f"Error during data validation step: {e}")
            traceback.print_exc()
            return
        
        while self.global_step.numpy() < self.tcfg['total_steps']:
            try:
                batch_data = next(self.train_iter, None)
                if batch_data is None: 
                    print("Training data iterator exhausted, recreating...")
                    self.train_iter = iter(self.train_ds)
                    batch_data = next(self.train_iter, None)
                if batch_data is None: 
                    print("Failed to get batch data after recreating iterator. Stopping.")
                    break
                us, pi, wdl, q, st_q = batch_data
                if tf.reduce_sum(tf.abs(pi)) < 1e-8:
                    consecutive_zero_batches += 1
                    print(f"Warning: Policy targets are all zeros, skipping batch (count: {consecutive_zero_batches})")
                    if consecutive_zero_batches >= max_zero_batches: 
                        print(f"ERROR: Got {consecutive_zero_batches} consecutive zero batches. Data pipeline is broken!")
                        break
                    continue
                else: 
                    consecutive_zero_batches = 0
                
                input_dict = {'us': us, 'pi': pi, 'wdl': wdl}
                losses = self.train_step(input_dict)
                step = self.global_step.numpy()
                
                if self.use_sparsity: 
                    self.pruning_callback.on_epoch_end(batch=step)
                
                log_steps = self.tcfg.get("train_avg_report_steps", 100)
                if step % log_steps == 0 and step > 0:
                    sps = log_steps / (time.time() - last_log_time) if time.time() > last_log_time else 0
                    last_log_time = time.time()
                    loss_str = ", ".join([f"{name.upper()}: {value.numpy():.6f}" for name, value in losses.items()])
                    inner_lr_attr = self.optimizer.inner_optimizer.learning_rate
                    if isinstance(inner_lr_attr, tf.keras.optimizers.schedules.LearningRateSchedule): 
                        current_lr = inner_lr_attr(step).numpy()
                    else: 
                        current_lr = inner_lr_attr.numpy()
                    print(f"Step {step:6d}, LR: {current_lr:.6f}, Losses: [ {loss_str} ], SPS: {sps:.2f}")
                
                checkpoint_steps = self.tcfg.get("checkpoint_steps", 1000)
                if step > 0 and step % checkpoint_steps == 0:
                    self.manager.save(checkpoint_number=step)
                    self.export_leela_weights(step)
                    test_losses = self.evaluate()
                    if test_losses:
                        test_loss_str = ", ".join([f"Test {name.upper()}: {value:.6f}" for name, value in test_losses.items()])
                        print(f"Step {step:6d}, Evaluation Results: [ {test_loss_str} ]")
            except Exception as e:
                print(f"Unexpected error at step {self.global_step.numpy()}: {e}")
                traceback.print_exc()
                continue
        print("Training finished.")
