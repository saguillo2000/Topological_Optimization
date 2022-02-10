from math import log
from typing import Type
from functools import partial

import tensorflow as tf
import numpy as np

from model.Metrics.Metric import Metric


# w r=k*r_0, qt_spheres ~= qt_spheres_0*k^-dim
# dim = (ln(qt_spheres_0)-ln(qt_spheres))/ln(k)

# it might also be worthwile to implement a lazy variation, i.e. that does not compute the entire distance matrix
#   this depends on whether the execution time of the actual minkowski computation
#   becomes an issue before or after the distance matrix size


def _greedy_covering_size(distance_matrix, radius):

    qt_points, _ = distance_matrix.shape
    is_covered = tf.constant(False,dtype=tf.bool, shape=(qt_points,))
    qt_balls = 0

    for current_index in range(qt_points):
        if not is_covered[current_index]:
            qt_balls += 1
            covered_by_current_ball = tf.math.less(distance_matrix[current_index], radius)
            is_covered = tf.logical_or(is_covered, covered_by_current_ball)

    return qt_balls

def minkowski_dimension_from_explicit_metric(distance_matrix: tf.Tensor,
                                             diameter
                                             ):

    def r_values():
        rad = diameter
        while rad > 0.01*diameter:
            rad *= 0.9
            yield rad

    covering_size = lambda radius : _greedy_covering_size(distance_matrix, radius)

    r_values = np.fromiter(r_values(), dtype=np.float64)
    neighbors_estimates = np.fromiter(map(covering_size, r_values), dtype=np.float64)

    qt_samples = len(r_values)

    dimension_estimates = np.zeros((qt_samples,qt_samples))
    for i,j in np.stack(np.triu_indices(qt_samples,k=1)).T:
        dimension_estimates[i,j] = (np.log(neighbors_estimates[j])-np.log(neighbors_estimates[i])) \
                                   /(np.log(r_values[i])-np.log(r_values[j]))

    return np.percentile(dimension_estimates.flatten(),95)


def minkowski_dimension_from_explicit_metric_old(distance_matrix : tf.Tensor,
                                             diameter
                                             ):

    def r_values():
        rad = 1
        while rad>0.01:
            rad *= 0.9
            yield rad

    dim_estimates = []
    covering_size_at_diam= _greedy_covering_size(distance_matrix,diameter)

    for r in r_values():
        covering_size = _greedy_covering_size(distance_matrix,r*diameter)
        dim_estimate = (log(covering_size_at_diam)-log(covering_size))/log(r)
        dim_estimates.append(dim_estimate)

    return max(dim_estimates)


def minkowski_dimension(points : tf.Tensor,
                        metric : Type[Metric]
                        ):

    distance_matrix = metric.distance_matrix(points)
    diameter = metric.diameter if metric.diameter != float("inf") else 1

    return minkowski_dimension_from_explicit_metric(distance_matrix, diameter)
