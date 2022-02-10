from functools import partial

import tensorflow as tf
import numpy as np
import os

from constraints import EPOCH_ITERATIONS_TO_INCREMENT_ITERATION_BATCHES
from model import NeuronSpace
from model.NeuronClusteringStrategies import AverageImportanceStrategy

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from ToyNeuralNetworks.datasets.CIFAR10.dataset import get_dataset


def fibonacci(initial_a=1, initial_b=1):
    a, b = initial_a, initial_b
    while True:
        yield a
        a = b
        a, b = b, a + b


def batches_generator(number_of_batches):
    train_dataset_list = list(train_dataset)
    current_batch = 0
    while True:
        epoch_changed = False
        train_batches = []
        until_idx = (current_batch + number_of_batches) % len(train_dataset_list)
        if until_idx <= current_batch:
            train_batches.extend(train_dataset_list[current_batch:])
            train_batches.extend(train_dataset_list[:until_idx])
            epoch_changed = True
        elif until_idx > current_batch:
            train_batches.extend(train_dataset_list[current_batch:until_idx])
        current_batch = until_idx
        yield train_batches, epoch_changed


def group_label(inputs, labels):
    groups = dict()

    for idx in range(len(labels)):
        label = str(int(labels[idx]))
        if label not in groups:
            groups[label] = []
        groups[label].append(inputs[idx])

    return groups


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_dataset()

    model = tf.keras.Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2), padding='same'),

        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2), padding='same'),

        Flatten(),
        Dense(units=256, activation='relu'),
        Dense(units=512, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')  # CIFAR 10
    ])

    batches_incrementation_strategy = partial(fibonacci, initial_a=34, initial_b=55)
    clustering_strategy = partial(AverageImportanceStrategy.average_importance_clustering, number_of_neurons=3000)
    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering,
                                    neuron_clustering_strategy=clustering_strategy)
    epochs = 1
    batches_incrementation_strategy = fibonacci(initial_a=34, initial_b=55)

    for epoch in range(epochs):

        number_of_batches = next(batches_incrementation_strategy)
        number_of_batches = min(len(train_dataset), number_of_batches)
        assert number_of_batches <= len(train_dataset)
        train_batches = batches_generator(number_of_batches)

        batches, changed_epoch = next(train_batches)

        for inputs, labels in batches:
            grouped_inputs = group_label(inputs, labels)

            for label, data_group in grouped_inputs.items():
                tf_label = tf.ones(len(data_group)) * int(label)
                tf_data = tf.convert_to_tensor(data_group)
                X = neuron_space_strategy(model, tf_data)
                print(X)

    print('Finished')
