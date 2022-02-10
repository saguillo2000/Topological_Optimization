import numpy as np

from model.MappedMatrix import MappedMatrix
from collections import Counter


def max_importance_clustering(activations_x_examples, number_of_neurons):
    def sample_neurons(activations_x_examples, neuron_idxs_counter):
        neurons_idxs = list(range(activations_x_examples.array.shape[0]))
        total_counter_neurons = len(list(neuron_idxs_counter))
        total_counter = activations_x_examples.array.shape[1]
        total_neurons = activations_x_examples.array.shape[0]
        difference_neurons_counter = total_neurons - total_counter_neurons
        probs = np.array(list(map(
            lambda neuron_idx: compute_probs(neuron_idx, neuron_idxs_counter, difference_neurons_counter,
                                             total_counter), neurons_idxs)))
        return np.random.choice(neurons_idxs, size=number_of_neurons, replace=False, p=probs)

    def compute_probs(neuron_idx, neuron_idxs_counter, difference_neurons_counter, total_counter):
        if neuron_idx in neuron_idxs_counter:
            return neuron_idxs_counter[neuron_idx] / (total_counter + 1)
        else:
            return (1 / (difference_neurons_counter * (total_counter + 1)))

    max_activation_neuron_idxs = []
    for idx_col in range(activations_x_examples.array.shape[1]):
        max_activation_neuron_idxs.append(np.argmax(np.abs(activations_x_examples.array[:, idx_col])))

    neuron_idxs_counter = Counter(max_activation_neuron_idxs)
    sampled_neurons = sample_neurons(activations_x_examples, neuron_idxs_counter)
    sampled_activations_x_examples = np.take(activations_x_examples.array, sampled_neurons, 0)
    return MappedMatrix(array=sampled_activations_x_examples)