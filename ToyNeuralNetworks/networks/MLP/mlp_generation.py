import itertools

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from networks.MLP.neuron_distributions import Pareto


def generate_networks(num_models: int,
                      input_shape: tuple, max_hidden_layers: int, output_layer_size: int, num_neurons: int,
                      generator=Pareto, parameter_generator=Pareto.default_parameter_distr):
    num_hidden_neurons = num_neurons - output_layer_size

    inflated_output_layer_size = output_layer_size * 1.3  # read the notes on NeuronDistribution.last_layer_restricted for clarification

    networks_for_generators = (_build_partial_mlps(input_shape,
                                                   generator \
                                                   .last_layer_restricted(max_hidden_layers,
                                                                          num_hidden_neurons,
                                                                          inflated_output_layer_size,
                                                                          *parameters) \
                                                   .distribute_neurons(max_hidden_layers,
                                                                       num_hidden_neurons),
                                                   output_layer_size)
                               for parameters in parameter_generator(num_models))

    networks = list(itertools.chain(*networks_for_generators))

    return networks


network_qt = 0


def _build_partial_mlps(input_shape: tuple, neurons_per_hidden_layer: np.ndarray, output_layer_size: int):
    # ¿i think? this was noticeably faster when networks were built on the fly (like in _generate_networks)

    networks = []
    for hidden_layer_qt in range(len(neurons_per_hidden_layer)):
        hidden_layer_subset = neurons_per_hidden_layer[:hidden_layer_qt + 1]
        networks.append(_build_mlp(input_shape, hidden_layer_subset, output_layer_size))
    global network_qt
    network_qt += len(networks)
    return networks


def _build_mlp(input_shape: tuple, neurons_per_hidden_layer: np.ndarray, output_layer_size: int):
    network = keras.Sequential()
    network.add(layers.Flatten(input_shape=input_shape))
    for layer_size in neurons_per_hidden_layer:
        network.add(layers.Dense(layer_size, activation='relu'))  # TODO random activation?
    network.add(layers.Dense(output_layer_size))

    return network


if __name__ == "__main__":

    qt_batches = 10  # "batches" here refers to a set of networks taken from pre-slicing a full network (_build_partial_mlps)
    input_shape = (32, 32, 3)
    max_hidden_layers = 6
    output_size = 10
    max_neurons = 1000

    networks = generate_networks(qt_batches,
                                 input_shape,
                                 max_hidden_layers,
                                 output_size,
                                 max_neurons)
    for network in networks:
        network.summary()
    print("====================")
    print("Quantity of networks: ")
    print(len(networks))
