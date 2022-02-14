from functools import partial

import tensorflow as tf
import numpy as np
import os

from gudhi.wasserstein import wasserstein_distance

from filtrations import RipsModel
from model import NeuronSpace
from model.NeuronClusteringStrategies import AverageImportanceStrategy
from tensorflow.keras import losses

from ToyNeuralNetworks.datasets.CIFAR10.dataset import get_dataset
from ToyNeuralNetworks.networks.MLP.mlp_generation import generate_networks


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


@tf.function
def _compute_predictions(inputs, model):
    return model(inputs)


if __name__ == '__main__':
    topo_reg = 0.3
    train_dataset, val_dataset, test_dataset = get_dataset()

    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)
    model = generate_networks(1, (32, 32, 3), 8, 10, 4000)[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    print(model.summary())
    '''
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
    '''

    batches_incrementation_strategy = partial(fibonacci, initial_a=34, initial_b=55)
    clustering_strategy = partial(AverageImportanceStrategy.average_importance_clustering, number_of_neurons=3000)
    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering,
                                    neuron_clustering_strategy=clustering_strategy)
    epochs = 50
    batches_incrementation_strategy = fibonacci(initial_a=34, initial_b=55)

    for epoch in range(epochs):

        # number_of_batches = next(batches_incrementation_strategy)
        # number_of_batches = min(len(train_dataset), number_of_batches)
        # assert number_of_batches <= len(train_dataset)
        number_of_batches = 1
        train_batches = batches_generator(number_of_batches)

        batches, changed_epoch = next(train_batches)

        for inputs, labels in batches:
            grouped_inputs = group_label(inputs, labels)

            for label, data_group in grouped_inputs.items():
                tf_label = tf.ones(len(data_group)) * int(label)
                # dim_label = tf.shape(tf_label).numpy()[0]
                # tf_label = tf.reshape(tf_label, [1, dim_label])

                tf_data = tf.convert_to_tensor(data_group)
                X = neuron_space_strategy(model, tf_data)

                X = tf.Variable(X.array, tf.float64)

                with tf.GradientTape() as tape:
                    Dg = RipsModel(X=X, mel=10, dim=0, card=10).call()
                    topo_loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)
                    predictions_point = _compute_predictions(inputs, model)
                    single_loss = loss_object(labels, predictions_point)
                    loss = topo_reg * topo_loss + (1 - topo_reg)*single_loss
                    print('Loss of label, ',tf_label, ' is :', loss)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print('------------------------------------')
    print('Finished')
