import tensorflow as tf


# The Haussdorff distance between two spaces is computed as:
# max {max_{x in A}min_{y in B}||x-y||, max_{x in B}min_{y in A}||x-y||}
# This function assumes that A and B are represented by a number of finite points in the same dimension (R^d)
# These representation will be given in a matrix which has as many rows as points and as many columns as the dimension
# of the containing space (d)
def discrete_haussdorff_distance(matrix_A, matrix_B, A_subspace_of_B=False):
    A_samples = matrix_A.shape[0]
    B_samples = matrix_B.shape[0]
    A_indices = range(A_samples)
    B_indices = range(B_samples)

    mins_second_term = tf.convert_to_tensor(
        list(map(lambda B_idx: _get_min_list(B_idx, A_indices, matrix_B, matrix_A), B_indices))
    )

    if A_subspace_of_B:
        return tf.math.reduce_max(mins_second_term)

    mins_first_term = tf.convert_to_tensor(
        list(map(lambda A_idx: _get_min_list(A_idx, B_indices, matrix_A, matrix_B), A_indices))
    )
    return tf.math.maximum(tf.math.reduce_max(mins_first_term), tf.math.reduce_max(mins_second_term))



def _get_min_list(fixed_idx, other_space_indices, fixed_space, other_space):
    fixed_sample = fixed_space[fixed_idx, :]
    distances_between_fixed_and_others = list(map(lambda other_idx: tf.norm(
        fixed_sample - other_space[other_idx, :]
    ), other_space_indices))
    return tf.math.reduce_min(distances_between_fixed_and_others)
