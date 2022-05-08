from functools import partial

from model import Network
from model.NeuronClusteringStrategies import RandomSampleStrategy


def get_full_neuron_activations_space(model, x_input):
    return get_neuron_activation_space_with_clustering(model, x_input, None)


def get_sampled_neuron_activations_space(model, x_input, number_of_neurons: int):
    neuron_clustering_strategy = partial(RandomSampleStrategy.random_sampling_clustering,
                                         number_of_neurons=number_of_neurons)
    return get_neuron_activation_space_with_clustering(model, x_input, neuron_clustering_strategy)


def get_neuron_activation_space_with_clustering(model, x_input, neuron_clustering_strategy=None):
    activations_x_examples = Network.get_neuron_activations_x_examples_matrix(x_input, model)
    if neuron_clustering_strategy is not None:
        original_activations_x_examples = activations_x_examples
        print(1)
        print()
        activations_x_examples = neuron_clustering_strategy(activations_x_examples)
        original_activations_x_examples.delete_matrix()
    return activations_x_examples


def get_neuron_activation_space_with_clustering_tf(model, x_input):
    activations_x_examples = Network.get_neuron_activations_x_examples_matrix_tf(x_input, model)
    return activations_x_examples
