#!/usr/bin/env python3
import gzip, os, numpy as np
import net_pb2 as pb

LC0_MAJOR, LC0_PATCH = 0, 0
LC0_MINOR_WITH_MAMBA2 = 33
WEIGHTS_MAGIC = 0x1c0

def nested_getattr(obj, attr):
    for a in attr.split('.'):
        obj = obj[int(a)] if a.isdigit() else getattr(obj, a)
    return obj

class Net:
    def __init__(self, net_fmt=pb.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT, **kwargs):
        self.pb = pb.Net(); self.pb.magic = WEIGHTS_MAGIC
        self.pb.min_version.major, self.pb.min_version.patch = LC0_MAJOR, LC0_PATCH
        self.pb.min_version.minor = LC0_MINOR_WITH_MAMBA2
        self.pb.format.weights_encoding = pb.Format.LINEAR16
        self.set_networkformat(net_fmt)
        for k, v in kwargs.items(): getattr(self, f"set_{k}")(v)

    def set_networkformat(self, net_fmt): self.pb.format.network_format.network = net_fmt
    def set_valueformat(self, v): self.pb.format.network_format.value = v; self.pb.format.network_format.output = pb.NetworkFormat.OUTPUT_WDL if v == pb.NetworkFormat.VALUE_WDL else pb.NetworkFormat.OUTPUT_CLASSICAL
    def __getattr__(self, name):
        if name.startswith("set_"):
            field = name[4:]
            def setter(v): setattr(self.pb.format.network_format, field, v)
            return setter
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def fill_layer(self, layer, params):
        params = params.flatten().astype(np.float32)
        if len(params) == 0: layer.min_val, layer.max_val, layer.params = 0, 0, b''; return
        layer.min_val, layer.max_val = float(np.min(params)), float(np.max(params))
        p_norm = (params - layer.min_val) / (layer.max_val - layer.min_val) if layer.max_val != layer.min_val else np.zeros_like(params)
        layer.params = np.round(p_norm * 0xffff).astype(np.uint16).tobytes()

    def tf_name_to_pb_path(self, tf_name):
        parts = tf_name.split('/')
        if not parts: return None

        # Keras layer names (e.g., "input/dense") become the initial part of the tf_name
        base_name = parts[0]
        # The actual weight name (e.g., "kernel:0") is the last part
        weight_type_full = parts[-1]
        weight_type_base = weight_type_full.split(':')[0]
        
        path = ['weights']

        # Handle "input/dense/kernel:0"
        if base_name == 'input':
            suffix = 'weights' if weight_type_base == 'kernel' else 'biases'
            # Maps to the ConvBlock 'input' in the proto.
            path.extend(['input', suffix])
            return '.'.join(path)

        # Handle "encoder_i/mamba2/..."
        elif base_name.startswith('encoder_'):
            block_num = base_name.split('_')[-1]
            block_type = parts[1] # e.g., 'mamba2'
            path.extend(['encoder', block_num, block_type])
            
            # Sub-path within the Mamba2 block
            sub_path_parts = parts[2:] # e.g., ['in_proj', 'kernel:0'] or ['A_log:0']
            
            if block_type == 'mamba2':
                # Case 1: Standalone weights like "A_log:0" or "D:0"
                if len(sub_path_parts) == 1:
                    pb_field_name = sub_path_parts[0].split(':')[0]
                    path.append(pb_field_name)
                    return '.'.join(path)
                
                # Case 2: Weights inside a Keras layer, e.g., "in_proj/kernel:0"
                elif len(sub_path_parts) == 2:
                    layer_name = sub_path_parts[0] # "in_proj", "conv1d", etc.
                    weight_type = sub_path_parts[1].split(':')[0] # "kernel" or "bias"
                    
                    if layer_name == 'norm':
                        path.append('norm.gammas')
                    else:
                        suffix = 'weights' if weight_type in ['kernel', 'gamma'] else 'biases'
                        pb_field_name = f"{layer_name}_{suffix}"
                        path.append(pb_field_name)
                    return '.'.join(path)

        # Handle "policy/dense/..." and "value/dense/..."
        elif base_name in ('policy', 'value'):
            is_value = (base_name == 'value')
            
            pb_heads_container = 'value_heads' if is_value else 'policy_heads'
            # For this model, map to the primary head for simplicity
            pb_head_name = 'winner' if is_value else 'vanilla'
            
            if is_value:
                # 'value/dense' -> 'ip1_val_w'/'ip1_val_b'
                pb_layer_prefix = 'ip1_val'
            else:
                # 'policy/dense' -> 'ip_pol_w'/'ip_pol_b'
                pb_layer_prefix = 'ip_pol'
            
            suffix = 'w' if weight_type_base == 'kernel' else 'b'
            path.extend([pb_heads_container, pb_head_name, f"{pb_layer_prefix}_{suffix}"])
            return '.'.join(path)
            
        return None

    def populate_from_tf_weights(self, tf_weights_dict):
        max_encoder_idx = -1
        for tf_name in tf_weights_dict:
            if tf_name.startswith('encoder_'):
                try:
                    idx = int(tf_name.split('/')[0].split('_')[-1])
                    if idx > max_encoder_idx: max_encoder_idx = idx
                except (ValueError, IndexError): pass
        
        if max_encoder_idx > -1:
            for _ in range(max_encoder_idx + 1): self.pb.weights.encoder.add()

        for tf_name, array in tf_weights_dict.items():
            pb_path = self.tf_name_to_pb_path(tf_name)
            if not pb_path:
                print(f"Warning: No mapping for TF weight '{tf_name}', skipping.")
                continue
            try:
                self.fill_layer(nested_getattr(self.pb, pb_path), array)
            except Exception as e:
                # Provide more context on failure
                print(f"ERROR: Failed to populate '{tf_name}' (mapped to '{pb_path}'). Reason: {e}")
                import traceback
                traceback.print_exc()


    def save_proto(self, filename):
        if not filename.endswith(".pb.gz"): filename += ".pb.gz"
        with gzip.open(filename, 'wb') as f: f.write(self.pb.SerializeToString())
        print(f"Weights saved as '{filename}' ({os.path.getsize(filename) / 1024**2:.2f} MB)")
