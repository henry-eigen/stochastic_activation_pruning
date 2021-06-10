import numpy as np
import tensorflow as tf

# input shape = (height, width)
def per_filter(acts, percent=0.16):
    
    # flattened input
    flat_acts = tf.reshape(acts, [-1])

    # abs(flattened input)
    flat_abs = tf.math.abs(flat_acts)

    # helper variables (input shape)
    input_length = tf.cast(tf.shape(flat_acts), tf.int64)
    num_selected = tf.cast(tf.cast(tf.shape(flat_acts), tf.float32) * percent, tf.int32)

    # get probability dist for activations (l1 for rescale, log for sampling)
    l1_probs = flat_abs / tf.reduce_sum(flat_abs)
    log_probs = tf.expand_dims(tf.math.log(flat_abs), 0)

    # get idx's of kept activations
    selected = tf.random.categorical(log_probs, tf.squeeze(num_selected))
    unique, _ = tf.unique(tf.squeeze(selected, 0))

    # values of kept activations
    h_vals = tf.gather(flat_acts, unique)

    # rescale probabilities of kept activations
    p_vals = tf.gather(l1_probs, unique)
    p_rescaled = 1 / (1 - tf.pow(1 - p_vals, tf.cast(num_selected, tf.float32)))

    # scatter h_vals into zeros_like activations
    h_acts = tf.scatter_nd(indices=tf.expand_dims(unique, -1),
                           updates=h_vals,
                           shape=input_length)

    # scatter p_vals into zeros_like activations
    p_acts = tf.scatter_nd(indices=tf.expand_dims(unique, -1),
                           updates=p_rescaled,
                           shape=input_length)

    return tf.reshape(h_acts * p_acts, tf.shape(acts))

# input shape = (height, width, channels)
def per_channel(x, percent):
    
    # change axes to be channel first since tf.fn_map always goes by axis 0
    x = tf.reshape(x, [tf.shape(x)[2], tf.shape(x)[0], tf.shape(x)[1]])
    
    # for each channel do ...
    x = tf.map_fn(lambda acts: per_filter(acts, percent), x)
    
    # change axes back to original order
    x = tf.reshape(x, [tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[0]])
    
    return x

# input shape = (samples, height, width, channels)
def SAP(x, percent):
    
    # for each sample do ...
    return tf.map_fn(lambda acts: per_channel(acts, percent), x)