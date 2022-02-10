import random
from typing import Type

import numpy as np
import tensorflow as tf

from Configuration.Constants import Random as RandomConstants
from model.Metrics.CompleteManifold import CompleteManifold


def frechet_k_means_clustering(activations_x_examples : tf.Tensor,
                               number_of_neurons,
                               metric : Type[CompleteManifold],
                               speed = 0.1,
                               max_norm = 1):
    """
    Close to k-means clustering, but since direct computation of a mean is not guaranteed,
    the strategy used is to shift each prototype downwards along the slope of the frechet variance wrt its voronoi cell
    (the global minimum of this variance, if it exists, would be the frechet mean)
    only one optimization step (gradient descent or newton) is performed between each computation of the cells
    one might want to perform more, or wait until convergence, to obtain something closer to k-means, but this incurs
    a lot more computation of distances (and it's likely both are equivalent by some form of diagonal argument)
    """
    random.seed(RandomConstants.DEFAULT_SEED)

    activations_x_examples = activations_x_examples.numpy()

    representatives = metric.sample(number_of_neurons)

    for _ in range(100):
        distances = metric.distance(representatives,activations_x_examples)
        cluster_indices = np.argmin(distances,axis=0)
        displacements = np.zeros(representatives.shape)
        is_empty = np.ones(representatives.shape[0],dtype=bool)
        for cluster_index,vector in zip(cluster_indices,activations_x_examples):
            direction = metric.log(representatives[cluster_index],vector)

            # gradient
            # displacements[cluster_index] += direction*np.linalg.norm(direction)
            # newton
            displacements[cluster_index] += direction.squeeze()

            is_empty[cluster_index] = False

        for i,(representative,displacement) in enumerate(zip(representatives,displacements)):
            displacement *= speed
            d_norm = np.linalg.norm(displacement)
            if d_norm>max_norm:
                displacement *= max_norm/d_norm
            representatives[i] = metric.exp(representative,displacement*speed)

        for i,empty in enumerate(is_empty):
            if empty:
                representatives[i] = metric.sample(1, seed=random.randint(0,1000000000))

    return tf.constant(representatives)
