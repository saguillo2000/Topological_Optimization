import numpy as np

from model.MappedMatrix import MappedMatrix


def random_sampling_clustering(activations_x_examples,
                               number_of_neurons,
                               ):
    qt_points, ambient_dim = activations_x_examples.array.shape
    indices = np.random.choice(qt_points, number_of_neurons, replace=False)
    activations_x_examples_sample = MappedMatrix(array=activations_x_examples.array[indices, :])
    return activations_x_examples_sample
