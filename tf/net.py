#!/usr/bin/env python3
import gzip, os, numpy as np
import net_pb2 as pb # MODIFIED IMPORT

LC0_MAJOR, LC0_PATCH = 0, 0
LC0_MINOR_WITH_MAMBA2 = 33
WEIGHTS_MAGIC = 0x1c0

def nested_getattr(obj, attr):
    for a in attr.split('.'): obj = obj[int(a)] if a.isdigit() else getattr(obj, a)
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
        base, w_type = parts[0], parts[-1].split(':')[0]
        path = ['weights']
        
        if base.startswith('input'):
            path.extend(['input', 'weights' if w_type == 'kernel' else 'biases'])
        elif base.startswith('encoder_'):
            block_num, block_type, layer_name = parts[0].split('_')[-1], parts[1], parts[2]
            path.extend(['encoder', block_num, block_type])
            
            if block_type == 'mamba2':
                if layer_name == 'norm': path.extend(['norm', 'gammas']); return '.'.join(path)
                suffix = 'weights' if w_type == 'kernel' else 'biases' if w_type == 'bias' else None
                if suffix: path.append(f"{layer_name}_{suffix}")
                else: path.append(layer_name)
                return '.'.join(path)
        elif base.startswith('policy'):
            path.extend(['policy_heads', 'vanilla', 'ip_pol_w' if w_type == 'kernel' else 'ip_pol_b'])
            return '.'.join(path)
        elif base.startswith('value'):
            path.extend(['value_heads', 'winner', 'ip1_val_w' if w_type == 'kernel' else 'ip1_val_b'])
            return '.'.join(path)
        return None

    def populate_from_tf_weights(self, tf_weights_dict):
        for tf_name_with_suffix, array in tf_weights_dict.items():
            tf_name = tf_name_with_suffix.split(':')[0]
            pb_path = self.tf_name_to_pb_path(tf_name)
            if not pb_path: print(f"Warning: No mapping for TF weight '{tf_name}', skipping."); continue
            path_parts = pb_path.split('.')
            if len(path_parts) > 2 and path_parts[1] == 'encoder':
                idx = int(path_parts[2])
                while len(self.pb.weights.encoder) <= idx: self.pb.weights.encoder.add()
            try: self.fill_layer(nested_getattr(self.pb, pb_path), array)
            except Exception as e: print(f"ERROR: Failed to populate '{tf_name}' (mapped to '{pb_path}'). Reason: {e}")

    def save_proto(self, filename):
        if not filename.endswith(".pb.gz"): filename += ".pb.gz"
        with gzip.open(filename, 'wb') as f: f.write(self.pb.SerializeToString())
        print(f"Weights saved as '{filename}' ({os.path.getsize(filename) / 1024**2:.2f} MB)")
