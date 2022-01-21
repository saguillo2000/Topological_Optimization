import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow_probability as tfp


def distance_corr(X: tf.Tensor) -> tf.Tensor:
    correlation = np.corrcoef(X.numpy())
    np.nan_to_num(correlation, copy=False)
    return tf.convert_to_tensor(correlation, np.float64)


def distance_corr_ruben(X):
    correlation = np.sqrt(1 - np.power(np.corrcoef(X), 2))
    np.nan_to_num(correlation, copy=False)
    return correlation


def euclidean_distance(X: tf.Tensor) -> tf.Tensor:
    euclidean = euclidean_distances(X, X)
    return tf.convert_to_tensor(euclidean, tf.float32)


def distance_corr_tf(X: tf.Tensor) -> tf.Tensor:
    return tf.math.sqrt(1 - tf.math.pow(tfp.stats.correlation(tf.transpose(X), sample_axis=0, event_axis=-1), 2))
