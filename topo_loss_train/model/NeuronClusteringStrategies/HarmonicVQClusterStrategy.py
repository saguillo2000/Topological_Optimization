from typing import Type

import numpy as np
import tensorflow as tf

from Configuration.Constants import Random as RandomConstants
from model.Metrics.CompleteManifold import CompleteManifold


def harmonic_vq_clustering(activations_x_examples : tf.Tensor,
                           number_of_neurons,
                           metric : Type[CompleteManifold]
                           ):

    activations_x_examples = activations_x_examples.numpy()

    representatives = metric.sample(number_of_neurons)

    np.random.seed(RandomConstants.DEFAULT_SEED)
    np.random.shuffle(activations_x_examples)

    for n,point in enumerate(activations_x_examples):
        # gradient of the harmonic mean of square distances (point-representatives), scaled by min_dist and offset by 1
        dist = metric.distance(point,representatives)
        min_dist = np.min(dist)
        sq_dist = np.square(dist)
        harmonic_error = 1+np.sum(np.divide(min_dist,sq_dist))
        weights = np.square(np.divide(min_dist,np.multiply(harmonic_error,np.square(sq_dist)))).T
        weights = np.sqrt(weights/np.max(weights))

        # some annealing smoothens results
        weights *= 0.5/(n+1)+0.5

        for i,(representative,weight) in enumerate(zip(representatives,weights)):
            representatives[i] = metric.combine(representative, point, weight)

    return tf.constant(representatives)
