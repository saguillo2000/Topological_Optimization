import math

import numpy as np
import tensorflow as tf

# Note that this only works if log_n > 1
from model import HaussdorffDistance


def get_confidence_set(activations_x_examples, alpha=0.05, maximum_N=50):
    n = activations_x_examples.shape[0]  # Number of samples
    activations_x_examples = tf.identity(activations_x_examples)
    b = int(np.ceil(n / np.log(n)))
    assert (b <= n)
    N = math.comb(n, b)
    N = min(N, maximum_N)
    L_b_alpha = 0
    for j in range(N):
        shuffled_activations = tf.random.shuffle(activations_x_examples)
        subspace = shuffled_activations[:b, :]
        T_j = HaussdorffDistance.discrete_haussdorff_distance(subspace, activations_x_examples)
        if T_j > alpha:
            L_b_alpha += T_j
    L_b_alpha = L_b_alpha/N
    return 2*np.power(L_b_alpha, -1)
