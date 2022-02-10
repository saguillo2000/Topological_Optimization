import numpy as np
from model.MappedMatrix import MappedMatrix
import time

def compute_distance_matrix(matrix_of_nodes):
    start = time.time()
    correlations = np.corrcoef(matrix_of_nodes.array)
    correlations = np.clip(np.nan_to_num(correlations, copy=False), -1, 1)
    np.fill_diagonal(correlations, 1)
    distance_matrix = 1-np.abs(correlations)
    end = time.time()
    print("Time to compute distance matrix: {}s".format(end-start))
    return MappedMatrix(array=distance_matrix)