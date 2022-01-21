import numpy as np
import tensorflow as tf


def distance_corr(X: tf.Tensor) -> tf.Tensor:
    correlation = np.corrcoef(X.numpy())
    np.nan_to_num(correlation, copy=False)
    return tf.convert_to_tensor(correlation, np.float64)


def distance_corr_ruben(X: tf.Tensor) -> tf.Tensor:
    correlation = np.sqrt(1 - np.power(np.corrcoef(X.numpy()), 2))
    np.nan_to_num(correlation, copy=False)
    return tf.convert_to_tensor(correlation, np.float64)
