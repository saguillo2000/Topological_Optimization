from typing import Type

import numpy as np
import tensorflow as tf

from model.Metrics.Metric import Metric


# this doesn't work very well - a better idea would be some sort of "cayler_menger_rank" (a very literal analogue)


def cayley_menger_determinant_from_explicit_metric(distance_matrix : tf.Tensor):
    qt_points, _ = distance_matrix.shape

    cm_matrix = np.pad(np.square(distance_matrix), ((0, 1), (0, 1)), 'constant', constant_values=1)
    cm_matrix[-1, -1] = 0
    cm_determinant = np.linalg.det(cm_matrix)*(-1 if qt_points%2 == 0 else 1)
    # in principle this is missing a factor of (qt_points+1)!^2 * 2^(qt_points+1), but this is constant wrt dimension

    return cm_determinant


def cayley_menger_determinant(points : tf.Tensor,
                              metric : Type[Metric]
                              ):

    distance_matrix = metric.distance_matrix(points)
    return cayley_menger_determinant_from_explicit_metric(distance_matrix)
