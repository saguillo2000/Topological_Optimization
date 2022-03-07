from functools import partial

import tensorflow as tf
import numpy as np

from gudhi.wasserstein import wasserstein_distance
from keras.models import clone_model

from filtrations import RipsModel
from model import NeuronSpace
from model.NeuronClusteringStrategies import AverageImportanceStrategy
from tensorflow.keras import losses

from ToyNeuralNetworks.datasets.CIFAR10.dataset import get_dataset

import pickle


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


def accuracy_model(name):
    return tf.metrics.SparseCategoricalAccuracy(name=name)


def labels_from_predictions(_predictions):
    return tf.expand_dims(tf.math.argmax(_predictions, 1), 1)


def train_step_topo(topo_reg, neuron_space_strategy, model, optimizer, loss_object, inputs, labels):
    X = neuron_space_strategy(model, inputs)  # Inputs all batch, with labels X = inputs
    X = tf.Variable(X.array, tf.float64)

    with tf.GradientTape() as tape:
        Dg = RipsModel(X=X, mel=10, dim=0, card=10).call()
        topo_loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)

        predictions_topo_reg = _compute_predictions(inputs, model)

        single_loss_topo_reg = loss_object(labels, predictions_topo_reg)

        loss_topo_reg = (topo_reg * topo_loss) + (1 - topo_reg) * single_loss_topo_reg

    gradients_topo = tape.gradient(loss_topo_reg, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_topo, model.trainable_variables))


def train_step(model, optimizer, loss_object, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = _compute_predictions(inputs, model)

        loss_none_topo_reg = loss_object(labels, predictions)

    gradients = tape.gradient(loss_none_topo_reg, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_experiment(epochs, topo_reg, accuracy_model, loss_object, model, optimizer,
                     train_dataset, val_dataset, neuron_space_strategy):
    model_topo_reg = clone_model(model)
    model_none_topo_reg = clone_model(model)

    for epoch in range(epochs):
        print('--------------------------------------')
        print('--------------------------------------')
        print('Epoch number:', epoch)
        print('--------------------------------------')
        print('--------------------------------------')

        number_of_batches = 20
        train_batches = batches_generator(number_of_batches)

        batches, changed_epoch = next(train_batches)

        for inputs, labels in batches:
            train_step_topo(topo_reg, neuron_space_strategy, model_topo_reg, optimizer, loss_object, inputs, labels)
            train_step(model_none_topo_reg, optimizer, loss_object, inputs, labels)

            predictions = _compute_predictions(inputs, model_none_topo_reg)
            predictions_topo_reg = _compute_predictions(inputs, model_topo_reg)
            print('Labels: ', labels)
            print('Predictions None reg: ', predictions)
            print('Predictions Reg: ', predictions_topo_reg)


if __name__ == '__main__':
    topo_reg = 0.3
    train_dataset, val_dataset, test_dataset = get_dataset()

    accuracy_model = accuracy_model(name='accuracy')

    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

    # generate_networks(1, (32, 32, 3), 8, 10, 4000)[0]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[32, 32, 3]),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam()

    batches_incrementation_strategy = partial(fibonacci, initial_a=34, initial_b=55)
    clustering_strategy = partial(AverageImportanceStrategy.average_importance_clustering, number_of_neurons=3000)
    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering,
                                    neuron_clustering_strategy=clustering_strategy)

    print('MODEL ARCHITECTURE FOR TOPO REG AND NONE TOPO REG: ')
    print(model.summary())

    train_experiment(20, topo_reg, accuracy_model, loss_object, model, optimizer,
                     train_dataset, val_dataset, neuron_space_strategy)
