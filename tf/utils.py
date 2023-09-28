import tensorflow as tf
import tensorflow_addons as tfa

def get_activation(activation):
    if activation == "mish":
        return tfa.activations.mish
    elif isinstance(activation, str) or activation is None:
        return tf.keras.activations.get(activation)
    else:
        return activation

def get_diffs(future):
    # future is a tensor of shape (batch_size, 1+n_future, 64, 12)
    # if the value does change over axis 1, keep the old value, else use 0
    # return a tensor of shape (batch_size, n_future, 64, 13)
    diff = future - tf.roll(future, shift=1, axis=1)
    changes = tf.abs(diff)
    out = changes * future
    out = out[:, 1:, :, :]
    return out

def make_one_sum(x):
    # append a final dimension so that the sum over the last axis is 1
    s = tf.sum(x, axis=-1, keepdims=True)
    x = tf.concat(x, 1 - s, axis=-1)
    return x

