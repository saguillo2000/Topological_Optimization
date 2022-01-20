import numpy as np
import tensorflow as tf


def distance_corr(X: np.array) -> tf.Tensor:
    correlation = np.corrcoef(X)
    np.nan_to_num(correlation, copy=False)
    return tf.convert_to_tensor(correlation, np.float64)


def distance_corr_ruben(X: np.array) -> tf.Tensor:
    correlation = 1 - np.power(np.corrcoef(X), 2)
    np.nan_to_num(correlation, copy=False)
    return tf.convert_to_tensor(correlation, np.float64)
