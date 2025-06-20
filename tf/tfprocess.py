#!/usr/bin/env python3
import numpy as np
import os, time, tensorflow as tf
import traceback
import net_pb2 as pb
from net import Net

# --- Keras Layers ---
class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(input_shape[-1],), initializer="ones", trainable=True)
    def call(self, x):
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        inv_rms = tf.math.rsqrt(tf.maximum(variance, self.eps))
        inv_rms = tf.where(tf.math.is_finite(inv_rms), inv_rms, tf.zeros_like(inv_rms))
        result = x * inv_rms * self.gamma
        return tf.where(tf.math.is_finite(result), result, x)

class RotaryPositionEmbedding2D(tf.keras.layers.Layer):
    def __init__(self, head_dim, theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.head_dim, self.theta = head_dim, float(theta)
    def build(self, input_shape):
        dim = self.head_dim
        indices = np.arange(0, self.head_dim, 2, dtype=np.float32)
        inv_freq = 1.0 / (self.theta ** (indices / self.head_dim))
        pos_x, pos_y = np.meshgrid(np.arange(8), np.arange(8))
        pos = np.stack([pos_y, pos_x], axis=-1).reshape(64, 2)
        freqs = np.clip(np.einsum('ij,k->ijk', pos, inv_freq).reshape(64, dim), -20.0, 20.0)
        cos_vals, sin_vals = np.cos(freqs), np.sin(freqs)
        self.cos_emb = self.add_weight(name="cos_emb", shape=(64, dim), initializer=tf.constant_initializer(cos_vals), trainable=False)
        self.sin_emb = self.add_weight(name="sin_emb", shape=(64, dim), initializer=tf.constant_initializer(sin_vals), trainable=False)
    def call(self, x):
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        x_rotated = tf.concat([-x[..., self.head_dim//2:], x[..., :self.head_dim//2]], axis=-1)
        result = x * self.cos_emb + x_rotated * self.sin_emb
        return tf.where(tf.math.is_finite(result), result, x)

class Mamba2Block(tf.keras.layers.Layer):
    def __init__(self, d_model, d_state, d_conv, expand, dt_rank, **kwargs):
        kwargs['dtype'] = tf.float32 
        super().__init__(**kwargs)
        self.d_i, self.d_s, self.dt_r, self.d_c, self.d_m = int(expand * d_model), min(d_state, 16), dt_rank, d_conv, d_model
    def build(self, input_shape):
        he_init = tf.keras.initializers.HeNormal(seed=42)
        sm_init = tf.keras.initializers.RandomNormal(stddev=0.01, seed=43)
        tiny_init = tf.keras.initializers.RandomNormal(stddev=0.001, seed=44)
        self.norm = RMSNorm(name="norm", dtype=tf.float32)
        self.in_proj = tf.keras.layers.Dense(self.d_i * 2, use_bias=False, name="in_proj", dtype=tf.float32, kernel_initializer=he_init)
        self.conv1d = tf.keras.layers.Conv1D(self.d_i, self.d_c, padding="causal", name="conv1d", activation=None, dtype=tf.float32, kernel_initializer=he_init)
        self.x_proj = tf.keras.layers.Dense(self.dt_r + self.d_s * 2, use_bias=False, name="x_proj", dtype=tf.float32, kernel_initializer=sm_init)
        self.dt_proj = tf.keras.layers.Dense(self.d_i, name="dt_proj", dtype=tf.float32, kernel_initializer=tiny_init, bias_initializer=tf.constant_initializer(-2.0))
        self.D = self.add_weight(name="D", shape=(self.d_i,), initializer=tf.constant_initializer(0.01), trainable=True)
        self.out_proj = tf.keras.layers.Dense(self.d_m, use_bias=False, name="out_proj", dtype=tf.float32, kernel_initializer=he_init)
        a_log_init = np.clip(-np.log(np.linspace(0.5, 2.0, self.d_s)), -5.0, 0.0)
        self.A_log = self.add_weight(name="A_log", shape=(self.d_i, self.d_s), initializer=tf.constant_initializer(np.tile(a_log_init, (self.d_i, 1))), trainable=True)
    def safe_softplus(self, x):
        return tf.nn.softplus(tf.clip_by_value(x, -20.0, 20.0))
    def ssm_manual_loop(self, x, dt, B, C):
        batch_size, seq_len, _ = tf.unstack(tf.shape(x))
        x, dt, B, C = [tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)) for t in [x, dt, B, C]]
        A = -tf.exp(tf.clip_by_value(self.A_log, -5.0, 0.0))
        dt_raw = self.dt_proj(dt)
        delta = tf.clip_by_value(self.safe_softplus(tf.clip_by_value(dt_raw, -5.0, 0.0)), 1e-8, 0.01)
        h = tf.zeros((batch_size, self.d_i, self.d_s), dtype=self.compute_dtype)
        outputs_ta = tf.TensorArray(dtype=self.compute_dtype, size=seq_len, clear_after_read=False)
        A_expanded = tf.expand_dims(A, 0)
        for t in tf.range(seq_len):
            delta_t, x_t, B_t, C_t = delta[:, t, :], x[:, t, :], B[:, t, :], C[:, t, :]
            delta_A_t = tf.exp(tf.clip_by_value(tf.expand_dims(delta_t, -1) * A_expanded, -10.0, 0.0))
            delta_B_u_t = tf.clip_by_value((tf.expand_dims(delta_t, -1) * tf.expand_dims(B_t, 1)) * tf.expand_dims(x_t, -1), -10.0, 10.0)
            h_new = tf.clip_by_value(delta_A_t * h + delta_B_u_t, -50.0, 50.0)
            h = tf.where(tf.math.is_finite(h_new), h_new, h * 0.9)
            y_t = tf.reduce_sum(h * tf.expand_dims(C_t, 1), axis=-1)
            outputs_ta = outputs_ta.write(t, tf.where(tf.math.is_finite(y_t), y_t, tf.zeros_like(y_t)))
        return tf.transpose(outputs_ta.stack(), perm=[1, 0, 2])
    def call(self, x):
        x_fp32 = tf.cast(x, tf.float32)
        residual = tf.where(tf.math.is_finite(x_fp32), x_fp32, tf.zeros_like(x_fp32))
        x_norm = self.norm(residual)
        x_proj = tf.where(tf.math.is_finite(self.in_proj(x_norm)), self.in_proj(x_norm), tf.zeros_like(self.in_proj(x_norm)))
        x_in, z = tf.split(x_proj, 2, axis=-1)
        x_conv = self.conv1d(x_in)
        x_conv = tf.nn.swish(tf.clip_by_value(tf.where(tf.math.is_finite(x_conv), x_conv, tf.zeros_like(x_conv)), -10.0, 10.0))
        x_proj_out = tf.where(tf.math.is_finite(self.x_proj(x_conv)), self.x_proj(x_conv), tf.zeros_like(self.x_proj(x_conv)))
        dt_in, B_in, C_in = tf.split(x_proj_out, [self.dt_r, self.d_s, self.d_s], axis=-1)
        y = self.ssm_manual_loop(x_conv, dt_in, B_in, C_in)
        D_safe = tf.where(tf.math.is_finite(self.D), self.D, tf.ones_like(self.D) * 0.01)
        y_skip = y + x_conv * D_safe
        z_activated = tf.nn.silu(tf.clip_by_value(tf.where(tf.math.is_finite(z), z, tf.zeros_like(z)), -10.0, 10.0))
        y_gated = tf.where(tf.math.is_finite(y_skip * z_activated), y_skip * z_activated, tf.zeros_like(y_skip * z_activated))
        output = tf.clip_by_value(self.out_proj(y_gated), -5.0, 5.0)
        output = tf.where(tf.math.is_finite(output), output, tf.zeros_like(output))
        result_fp32 = tf.where(tf.math.is_finite(residual + output), residual + output, residual)
        return tf.cast(result_fp32, x.dtype)

# --- Main Training Process Class ---
class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mcfg = cfg["model"]
        self.tcfg = cfg["training"]
        self.net_type = self.mcfg.get("network", "attention_body")
        self.strategy = None
        self.root_dir = os.path.join(self.tcfg["path"], self.cfg["name"])
        net_kwargs = {'valueformat': pb.NetworkFormat.VALUE_WDL if self.mcfg.get('value') == 'wdl' else pb.NetworkFormat.VALUE_CLASSICAL}
        if self.net_type == 'mamba2':
            net_kwargs['net_fmt'] = pb.NetworkFormat.NETWORK_MAMBA2_WITH_HEADFORMAT
            for p in ['d_state', 'd_conv', 'expand_factor', 'num_heads', 'head_dim', 'chunk_size', 'dt_rank', 'use_norm', 'rope_2d_enabled', 'rope_2d_theta']:
                param_name = f"mamba2_{p}" if p not in ['rope_2d_enabled', 'rope_2d_theta'] else p
                if param_name in self.mcfg: net_kwargs[param_name] = self.mcfg[param_name]
        self.net = Net(**net_kwargs)
        tf.keras.mixed_precision.set_global_policy('float32')

    def init(self, train_ds, test_ds, validation_dataset=None):
        self.train_iter, self.test_iter = iter(train_ds), iter(test_ds)
        inputs = tf.keras.Input(shape=(112, 8, 8), name="input_planes")
        outputs = self.construct_net(inputs)
        self.model = tf.keras.Model(inputs, outputs)
        print(f"Model '{self.mcfg['network']}' created with {self.model.count_params():,} parameters.")
        lr = min(self.tcfg.get("learning_rate", 0.001), 0.0001)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=0.5)
        self.weight_decay = 1e-5
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, global_step=self.global_step)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.root_dir, max_to_keep=5, checkpoint_name=self.cfg["name"])

    def restore(self):
        if self.manager.latest_checkpoint:
            print(f"Restoring from {self.manager.latest_checkpoint}...")
            self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
            print(f"Restore complete. Global step is {self.global_step.numpy()}.")
        else:
            print("No checkpoint found, starting from scratch.")

    def save_leelaz_weights(self, filename):
        print("Saving Leela-style weights...")
        
        # Create weight mappings for the protobuf format
        weight_mappings = self.create_weight_mappings()
        weights_to_process = {}
        
        # Collect all weights, EXCLUDING RoPE weights which are not saved.
        for w in self.model.weights:
            # RoPE embeddings are fixed and recalculated by the engine. Do not save them.
            if 'rotary_position_embedding2d' in w.name:
                print(f"  Skipping non-saved weight: '{w.name}'")
                continue
            weights_to_process[w.name] = w.numpy()
            
        # Apply custom mappings first
        for tf_name, pb_setter in weight_mappings.items():
            if tf_name in weights_to_process:
                try:
                    pb_setter(weights_to_process[tf_name])
                    print(f"  Mapped '{tf_name}' to protobuf structure")
                    # Remove from dict so it's not processed by legacy parser
                    del weights_to_process[tf_name]
                except Exception as e:
                    print(f"  WARNING: Failed to map '{tf_name}': {e}. Will use legacy parser.")
        
        # Use legacy parser for remaining weights (Mamba blocks, heads, etc.)
        if weights_to_process:
            print(f"  Using legacy parser for {len(weights_to_process)} remaining weights...")
            try:
                self.net.populate_from_tf_weights(weights_to_process)
            except Exception as e:
                print(f"  WARNING: Legacy parser failed: {e}")
        
        # Save the protobuf
        self.net.save_proto(filename) # save_proto now prints its own success message
        print("Weight saving complete.")


    def create_weight_mappings(self):
        """
        Create mappings from TensorFlow weight names to protobuf setters.
        This handles weights that the legacy parser in net.py doesn't know about.
        """
        mappings = {}
        
        # The input layer `input/dense` maps to the `input` ConvBlock in the proto.
        # We need to map its kernel and bias to the `weights` and `biases` fields
        # within that ConvBlock. We use the `fill_layer` helper from the Net class.
        
        def set_input_weights(value):
            target_pb_layer = self.net.pb.weights.input.weights
            self.net.fill_layer(target_pb_layer, value)
            
        def set_input_biases(value):
            target_pb_layer = self.net.pb.weights.input.biases
            self.net.fill_layer(target_pb_layer, value)

        mappings['input/dense/kernel:0'] = set_input_weights
        mappings['input/dense/bias:0'] = set_input_biases
        
        # RoPE weights ('rotary_position_embedding2d/*') are NOT mapped because
        # they are not present in the protobuf schema and are meant to be
        # regenerated by the engine from config parameters.
        
        return mappings

    def construct_net(self, inputs):
        mcfg = self.mcfg
        flow = tf.transpose(tf.cast(inputs, tf.float32), [0, 2, 3, 1])
        flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])
        embedding_size = mcfg['embedding_size']
        flow = tf.keras.layers.Dense(embedding_size, name="input/dense", activation=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=44))(flow)
        flow = tf.nn.relu(tf.clip_by_value(flow, 0.0, 10.0))
        if self.net_type == 'mamba2':
            if mcfg.get('rope_2d_enabled', False):
                r_in = tf.reshape(flow, (-1, 64, mcfg['mamba2_num_heads'], mcfg['mamba2_head_dim']))
                r_out = RotaryPositionEmbedding2D(mcfg['mamba2_head_dim'], mcfg.get('rope_2d_theta', 10000.0), name='rotary_position_embedding2d')(tf.transpose(r_in, [0, 2, 1, 3]))
                flow = tf.reshape(tf.transpose(r_out, [0, 2, 1, 3]), (-1, 64, embedding_size))
            for i in range(mcfg['encoder_layers']):
                flow_prev = flow
                flow = Mamba2Block(embedding_size, mcfg['mamba2_d_state'], mcfg['mamba2_d_conv'], mcfg['mamba2_expand_factor'], mcfg['mamba2_dt_rank'], name=f"encoder_{i}/mamba2")(flow)
                flow = tf.where(tf.math.is_finite(flow), flow, flow_prev * 0.9)
        
        # --- FIXED SECTION ---
        # 1. Instantiate the layer ONCE
        pooling_layer = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")
        
        # 2. Call the layer on the input tensor to get the output
        pooled_flow_tensor = pooling_layer(flow)
        
        # 3. Use the TENSOR (not the layer call) in your logic for NaN safety
        pooled_flow = tf.where(tf.math.is_finite(pooled_flow_tensor), pooled_flow_tensor, tf.zeros_like(pooled_flow_tensor))

        policy_logits = tf.keras.layers.Dense(1858, name="policy/dense", kernel_initializer=tf.keras.initializers.HeNormal(seed=45))(pooled_flow)
        value_logits = tf.keras.layers.Dense(3 if mcfg['value'] == 'wdl' else 1, name="value/dense", kernel_initializer=tf.keras.initializers.HeNormal(seed=46))(pooled_flow)
        return {'policy': tf.cast(policy_logits, tf.float32), 'value_winner': tf.cast(value_logits, tf.float32)}

    def safe_cross_entropy(self, labels, logits):
        labels = tf.nn.relu(labels)
        label_sum = tf.reduce_sum(labels, axis=-1, keepdims=True)
        labels = tf.where(label_sum > 1e-8, labels / (label_sum + 1e-8), tf.ones_like(labels) / tf.cast(tf.shape(labels)[-1], labels.dtype))
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=tf.clip_by_value(logits, -20.0, 20.0))

    @tf.function
    def train_step(self, x, y, z):
        x, y, z = [tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)) for t in [x,y,z]]
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            policy_pred = tf.where(tf.math.is_finite(predictions['policy']), predictions['policy'], tf.zeros_like(predictions['policy']))
            value_pred = tf.where(tf.math.is_finite(predictions['value_winner']), predictions['value_winner'], tf.zeros_like(predictions['value_winner']))
            policy_loss = tf.reduce_mean(self.safe_cross_entropy(y, policy_pred))
            value_loss = tf.reduce_mean(self.safe_cross_entropy(z, value_pred))
            policy_loss = tf.where(tf.math.is_finite(policy_loss), policy_loss, 10.0) 
            value_loss = tf.where(tf.math.is_finite(value_loss), value_loss, 1.0)
            total_loss = policy_loss + value_loss
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables if 'bias' not in v.name and 'norm' not in v.name])
            total_loss += self.weight_decay * tf.where(tf.math.is_finite(l2_loss), l2_loss, 0.0)
            total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, 11.0)
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        safe_grads = [(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)), v) for g, v in zip(gradients, self.model.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(safe_grads)
        return total_loss, policy_loss, value_loss

    def process_loop(self, total_batch_size, num_evals, batch_splits):
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs("leelalogs", exist_ok=True)
        print(f"Starting training for {self.tcfg['total_steps']} steps...")
        consecutive_nan_count, max_consecutive_nans = 0, 5
        last_log_time = time.time()
        while self.global_step.numpy() < self.tcfg['total_steps']:
            try:
                batch_data = next(self.train_iter, None)
                if batch_data is None: print("Training data iterator exhausted. Stopping."); break
                x, y, z = batch_data[0], batch_data[1], batch_data[2]
                total_loss, policy_loss, value_loss = self.train_step(x, y, z)
                total_loss_val, policy_loss_val, value_loss_val = total_loss.numpy(), policy_loss.numpy(), value_loss.numpy()
                if np.isnan(total_loss_val) or np.isinf(total_loss_val):
                    consecutive_nan_count += 1
                    print(f"Step {self.global_step.numpy():6d}, NaN/Inf detected in loss (count: {consecutive_nan_count}). Skipping step.")
                    if consecutive_nan_count >= max_consecutive_nans: print("Too many consecutive NaN losses, stopping training."); self.validate_model_weights(); break
                    continue
                else: consecutive_nan_count = 0
                self.global_step.assign_add(1)
                current_step = self.global_step.numpy()
                if current_step % 100 == 0:
                    end_time = time.time()
                    sps = 100 / (end_time - last_log_time)
                    last_log_time = end_time
                    print(f"Step {current_step:6d}, Loss: {total_loss_val:.4f} (P: {policy_loss_val:.4f}, V: {value_loss_val:.4f}), SPS: {sps:.2f}")
                if current_step > 0 and current_step % self.tcfg.get("checkpoint_steps", 1000) == 0:
                    save_path = self.manager.save(checkpoint_number=current_step)
                    print(f"Checkpoint saved to {save_path}")
                    weights_filename = f"{self.root_dir}/{self.cfg['name']}-{current_step}.pb.gz"
                    self.save_leelaz_weights(weights_filename)
            except (tf.errors.InvalidArgumentError, tf.errors.OpError) as e:
                print(f"TensorFlow error at step {self.global_step.numpy()}: {e}. Attempting to skip batch."); self.validate_model_weights(); continue
            except Exception as e:
                print(f"Unexpected error at step {self.global_step.numpy()}: {e}"); traceback.print_exc(); print("Attempting to continue..."); continue
        print("Training finished.")
        current_step = self.global_step.numpy()
        final_save_path = self.manager.save(checkpoint_number=current_step)
        print(f"Final checkpoint saved to {final_save_path}")
        final_weights_filename = f"{self.root_dir}/{self.cfg['name']}-{current_step}.pb.gz"
        self.save_leelaz_weights(final_weights_filename)

    def validate_model_weights(self):
        print("Validating model weights for NaN/Inf values...")
        nan_weights, inf_weights = [], []
        for weight in self.model.weights:
            if np.any(np.isnan(weight.numpy())): nan_weights.append(weight.name)
            if np.any(np.isinf(weight.numpy())): inf_weights.append(weight.name)
        if nan_weights: print(f"WARNING: NaN values found in weights: {nan_weights}")
        if inf_weights: print(f"WARNING: Inf values found in weights: {inf_weights}")
        if not nan_weights and not inf_weights: print("All weights are finite.")

