from typing import Type

import numpy as np
import tensorflow as tf

from Configuration.Constants import Random as RandomConstants
from model.Metrics.CompleteManifold import CompleteManifold


# this is arbitrary
def _decay():
    return np.linspace(1, 0.1, num=1000000)


def basic_vq_clustering(activations_x_examples : tf.Tensor,
                        number_of_neurons,
                        metric : Type[CompleteManifold],
                        sensitivity_delta=0.001):

    activations_x_examples = activations_x_examples.numpy()

    representatives = metric.sample(number_of_neurons)

    sensitivities = np.zeros((number_of_neurons,))

    np.random.seed(RandomConstants.DEFAULT_SEED)
    np.random.shuffle(activations_x_examples)

    for value,vector in zip(_decay(),activations_x_examples):
        index = np.argmin(metric.distance(vector,representatives)-sensitivities)
        selected_representative = representatives[index]
        representatives[index] = metric.combine(selected_representative, vector, value)
        sensitivities[index] = 0
        sensitivities += sensitivity_delta

    return tf.constant(representatives)
