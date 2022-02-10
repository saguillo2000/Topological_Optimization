import numpy as np
from model.MappedMatrix import MappedMatrix
from multiprocessing import Pool


def compute_distance_matrix(matrix_of_nodes):
    number_of_nodes = matrix_of_nodes.array.shape[0]
    with Pool() as p:
        distance_matrix = sum(p.starmap(_compute_distance_for_row,
                                        list(map(lambda x: (matrix_of_nodes.array, x), range(number_of_nodes)))))
        return MappedMatrix(array=distance_matrix)


def _compute_distance_for_row(matrix_of_nodes, row_idx):
    number_of_nodes = matrix_of_nodes.shape[0]
    aux_matrix = np.zeros(shape=(number_of_nodes, number_of_nodes))
    j = 0
    while j <= row_idx:
        if row_idx == j:
            aux_matrix[row_idx, j] = 0
        else:
            correlation_between_nodes = np.clip(np.nan_to_num(np.corrcoef(matrix_of_nodes[row_idx, :],
                                                                          matrix_of_nodes[j, :])[0, 1],
                                                              copy=True), -1, 1)
            pseudosemimetric = 1 - np.abs(correlation_between_nodes)
            aux_matrix[row_idx, j] = pseudosemimetric
            aux_matrix[j, row_idx] = pseudosemimetric
        j += 1
    return aux_matrix
