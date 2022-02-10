from functools import partial

import numpy as np

# It returns a distance matrix strategy using a pure metric that only depends on the two vectors.
from model.MappedMatrix import MappedMatrix


def get_metric_strategy(metric):
    return partial(_compute_distance_matrix, metric=metric)


def _compute_distance_matrix(matrix_of_nodes, metric):
    number_of_nodes = matrix_of_nodes.shape[0]
    distance_matrix = MappedMatrix(shape=(number_of_nodes, number_of_nodes))
    for i in range(number_of_nodes):
        j = 0
        while j <= i:
            if i == j:
                distance_matrix.array[i, j] = 0
            else:
                distance = metric(matrix_of_nodes.array[i, :], matrix_of_nodes.array[j, :])
                distance_matrix.array[i, j] = distance
                distance_matrix.array[j, i] = distance
            j += 1
    return distance_matrix
