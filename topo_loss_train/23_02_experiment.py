import sys
from functools import partial

import tensorflow as tf
import numpy as np
import os

from gudhi.wasserstein import wasserstein_distance
from keras.models import clone_model

from distance_strategies.correlation_strategy import distance_corr_tf
from filtrations import RipsModel
from model import NeuronSpace
from model.NeuronClusteringStrategies import AverageImportanceStrategy
from tensorflow.keras import losses

from ToyNeuralNetworks.datasets.CIFAR10.dataset import get_dataset

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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


def accuracy_per_class(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix.diagonal() / matrix.sum(axis=1)


@tf.function
def _compute_predictions(inputs, model):
    return model(inputs)


def accuracy_model(name):
    return tf.metrics.SparseCategoricalAccuracy(name=name)


def labels_from_predictions(_predictions):
    return tf.expand_dims(tf.math.argmax(_predictions, 1), 1)


if __name__ == '__main__':
    topo_reg = 0.3
    train_dataset, val_dataset, test_dataset = get_dataset()

    # Functions for training and pick model

    accuracy_model_topo = accuracy_model(name='accuracy_topo')
    accuracy_model_none_topo = accuracy_model(name='accuracy_none_topo')
    accuracy_model_val_topo = accuracy_model(name='accuracy_topo')
    accuracy_model_val_none_topo = accuracy_model(name='accuracy_none_topo')

    loss_object_topo = losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_object_none_topo = losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_object_val_topo = losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_object_val_none_topo = losses.SparseCategoricalCrossentropy(from_logits=True)

    # generate_networks(1, (32, 32, 3), 8, 10, 4000)[0]
    model_topo_reg = tf.keras.list_models.Sequential([
        tf.keras.layers.Flatten(input_shape=[32, 32, 3]),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model_none_topo_reg = tf.keras.list_models.Sequential([
        tf.keras.layers.Flatten(input_shape=[32, 32, 3]),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer_topo = tf.keras.optimizers.Adam()
    optimizer_none_topo = tf.keras.optimizers.Adam()

    print('MODEL ARCHITECTURE FOR TOPO REG AND NONE TOPO REG: ')
    print(model_topo_reg.summary())

    batches_incrementation_strategy = partial(fibonacci, initial_a=34, initial_b=55)
    clustering_strategy = partial(AverageImportanceStrategy.average_importance_clustering, number_of_neurons=3000)
    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering,
                                    neuron_clustering_strategy=clustering_strategy)
    epochs = 20

    losses_epochs = []
    topo_losses_epochs = []
    topo_accuracies_epochs = []
    accuracies_epochs = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(epochs):
        print('--------------------------------------')
        print('--------------------------------------')
        print('Epoch number:', epoch)
        print('--------------------------------------')
        print('--------------------------------------')

        losses_batches = []
        topo_losses_batches = []

        accuracy_batches = []
        accuracy_topo_batches = []

        # number_of_batches = next(batches_incrementation_strategy)
        # number_of_batches = min(len(train_dataset), number_of_batches)
        # assert number_of_batches <= len(train_dataset)
        number_of_batches = 20
        train_batches = batches_generator(number_of_batches)

        batches, changed_epoch = next(train_batches)

        for inputs, labels in batches:
            X = neuron_space_strategy(model_topo_reg, inputs)  # Inputs all batch, with labels X = inputs

            X = tf.Variable(X.array, tf.float64)

            with tf.GradientTape() as tape:
                Dg = RipsModel(X=X, mel=10, dim=0, card=10).call()
                topo_loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)

                predictions_topo_reg = _compute_predictions(inputs, model_topo_reg)

                single_loss_topo_reg = loss_object_topo(labels, predictions_topo_reg)

                loss_topo_reg = (topo_reg * topo_loss) + (1 - topo_reg) * single_loss_topo_reg

            gradients_topo = tape.gradient(loss_topo_reg, model_topo_reg.trainable_variables)
            optimizer_topo.apply_gradients(zip(gradients_topo, model_topo_reg.trainable_variables))

            with tf.GradientTape() as tape:
                predictions = _compute_predictions(inputs, model_none_topo_reg)

                loss_none_topo_reg = loss_object_none_topo(labels, predictions)

            gradients = tape.gradient(loss_none_topo_reg, model_none_topo_reg.trainable_variables)
            optimizer_none_topo.apply_gradients(zip(gradients, model_none_topo_reg.trainable_variables))

            loss_batch = loss_none_topo_reg.numpy()
            topo_loss_ = loss_topo_reg.numpy()

            labels_prediction = labels_from_predictions(predictions)
            labels_prediction_topo = labels_from_predictions(predictions_topo_reg)

            print('Labels: ', labels)
            print('Predictions None reg: ', labels_prediction)
            print('Predictions Reg: ', labels_prediction_topo)

            # classification_report(labels, labels_prediction, zero_division=0, output_dict=True)
            accuracy_batch = accuracy_per_class(labels, labels_prediction)
            accuracy_topo = accuracy_per_class(labels, labels_prediction_topo)
            print('------------------------------------')
            print('Loss with Topo Reg :', topo_loss_)
            print('Loss without Topo Reg: ', loss_batch)
            print('Accuracy with Topo Reg :\n', accuracy_topo)
            print('Accuracy without Topo Reg: \n', accuracy_batch)
            print('------------------------------------')

            topo_losses_batches.append(topo_loss_)  # 10 labels
            losses_batches.append(loss_batch)

            accuracy_batches.append(accuracy_batch)
            accuracy_topo_batches.append(accuracy_topo)

        val_losses_epoch_none_topo = []
        val_losses_epoch_topo = []

        val_accuracies_topo = []
        val_accuracies_none_topo = []

        for validation_inputs, validation_labels in val_dataset:
            pred = _compute_predictions(validation_inputs, model_topo_reg)
            val_loss_topo = loss_object_val_topo(validation_labels, pred).numpy()
            val_accuracy_topo = accuracy_per_class(validation_labels,
                                                   labels_from_predictions(pred))

            pred = _compute_predictions(validation_inputs, model_none_topo_reg)
            val_loss_none_topo = loss_object_val_none_topo(validation_labels, pred).numpy()
            val_accuracy_none_topo = accuracy_per_class(validation_labels,
                                                        labels_from_predictions(pred))

            val_losses_epoch_none_topo.append(val_loss_none_topo)
            val_losses_epoch_topo.append(val_loss_topo)

            val_accuracies_topo.append(val_accuracy_topo)
            val_accuracies_none_topo.append(val_accuracy_none_topo)

        validation_losses.append((val_losses_epoch_none_topo, val_losses_epoch_topo))
        validation_accuracies.append((val_accuracies_none_topo, val_accuracies_topo))
        topo_losses_epochs.append(topo_losses_batches)
        losses_epochs.append(losses_batches)
        topo_accuracies_epochs.append(accuracy_topo_batches)
        accuracies_epochs.append(accuracy_batches)

    print('Losses epochs: ', losses_epochs)
    outputFile = open('losses_epochs.pkl', 'wb')
    pickle.dump(losses_epochs, outputFile)
    outputFile.close()

    print('Topological losses epochs: ', topo_losses_epochs)
    outputFile = open('topo_losses_epochs.pkl', 'wb')
    pickle.dump(topo_losses_epochs, outputFile)
    outputFile.close()

    print('Topological accuracies epochs: ', topo_accuracies_epochs)
    outputFile = open('topo_accuracies_epochs.pkl', 'wb')
    pickle.dump(topo_accuracies_epochs, outputFile)
    outputFile.close()

    print('Topological losses epochs: ', accuracies_epochs)
    outputFile = open('accuracies_epochs.pkl', 'wb')
    pickle.dump(accuracies_epochs, outputFile)
    outputFile.close()

    print('Validations losses epochs: ', validation_losses)
    outputFile = open('validations_losses.pkl', 'wb')
    pickle.dump(validation_losses, outputFile)
    outputFile.close()

    print('Validations accuracies epochs: ', validation_accuracies)
    outputFile = open('validations_accuracies.pkl', 'wb')
    pickle.dump(validation_accuracies, outputFile)
    outputFile.close()

    print('Finished')
