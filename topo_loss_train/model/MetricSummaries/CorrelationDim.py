from enum import Enum
from functools import partial
from typing import Type

import numpy as np
import tensorflow as tf

from model.Metrics.Metric import Metric


# qn(r) := (qt entries in dist matrix that do not surpass r)/(qt points) = average qt pts within r for every point
# qn(r*D) ~= qn(D) * r^dim
# dim ~= (ln(qn(r*D)) - ln(qn(D)))/ln(r)


def _average_qt_pairs_within_r(r,distance_matrix):
    return tf.math.count_nonzero(tf.less(distance_matrix, r))/distance_matrix.shape[0]


def _median_qt_pairs_within_r(r,distance_matrix):
    neighbor_qts = tf.math.count_nonzero(tf.less(distance_matrix, r),axis=0).numpy()
    return np.median(neighbor_qts)


class NeighborMeasuringStrategy(Enum):
    AVERAGE = 1
    MEDIAN = 2


def correlation_dimension_from_explicit_metric(distance_matrix: tf.Tensor,
                                               diameter,
                                               neighbor_measuring_strategy=NeighborMeasuringStrategy.AVERAGE):
    if neighbor_measuring_strategy == NeighborMeasuringStrategy.AVERAGE:
        count_neighbors = partial(_average_qt_pairs_within_r, distance_matrix=distance_matrix)
    elif neighbor_measuring_strategy == NeighborMeasuringStrategy.MEDIAN:
        count_neighbors = partial(_median_qt_pairs_within_r, distance_matrix=distance_matrix)
    else:
        raise Exception("Invalid neighbor measuring strategy for correlation dimension computation")

    def r_values():
        rad = diameter
        while rad > 0.01*diameter:
            rad *= 0.9
            yield rad

    r_values = np.fromiter(r_values(), dtype=np.float64)
    neighbors_estimates = np.fromiter(map(count_neighbors, r_values), dtype=np.float64)

    qt_samples = len(r_values)

    dimension_estimates = np.zeros((qt_samples,qt_samples))
    for i,j in np.stack(np.triu_indices(qt_samples,k=1)).T:
        dimension_estimates[i,j] = (np.log(neighbors_estimates[i])-np.log(neighbors_estimates[j])) \
                                   /(np.log(r_values[i])-np.log(r_values[j]))

    return np.max(dimension_estimates)


def correlation_dimension(points : tf.Tensor,
                          metric : Type[Metric],
                          neighbor_measuring_strategy=NeighborMeasuringStrategy.AVERAGE):

    distance_matrix = metric.distance_matrix(points)
    diameter = metric.diameter if metric.diameter != float("inf") else 1

    return correlation_dimension_from_explicit_metric(distance_matrix,diameter,
                                                      neighbor_measuring_strategy=neighbor_measuring_strategy)