import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow_probability as tfp


def euclidean_distance(X: tf.Tensor) -> tf.Tensor:
    euclidean = euclidean_distances(X, X)
    return tf.convert_to_tensor(euclidean, tf.float32)


def correlation_numpy(x):
    return np.sqrt(1 - np.power(np.corrcoef(x), 2))


def replace_nan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)


def distance_corr_tf(x):
    y = tf.math.sqrt(1 - tf.math.pow(tfp.stats.correlation(tf.transpose(x), sample_axis=0, event_axis=-1), 2))
    # return tf.numpy_function(correlation_numpy, [input], tf.float32)
    return replace_nan(y)
