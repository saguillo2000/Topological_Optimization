# Adapter para trabajar con modelo de tensor flow

import numpy as np
import tensorflow as tf

from model.MappedMatrix import MappedMatrix


# Accuracy function is a function that, given a tensorflow dataset and a tensorflow model, it returns an accuracy
def get_generalization_gap(train_dataset, test_dataset, models, accuracy_fn):
    test_accuracies = list(map(lambda model: accuracy_fn(test_dataset, model), models))
    train_accuracies = list(map(lambda model: accuracy_fn(train_dataset, model), models))
    return np.array(list(map(lambda test_x_train: test_x_train[1] - test_x_train[0],
                             zip(test_accuracies, train_accuracies))))


def get_neuron_activations_x_examples_matrix(x, model, num_skipped_layers_from_start=1):
    examples_x_activations = _examples_x_activations_for_input(x, model, num_skipped_layers_from_start)
    activations_x_examples = examples_x_activations.transpose()
    examples_x_activations.delete_matrix()
    return activations_x_examples


def get_neuron_activations_x_examples_matrix_tf(x, model, num_skipped_layers_from_start=1):
    examples_x_activations = _examples_x_activations_for_input_tf(x, model, num_skipped_layers_from_start)
    activations_x_examples = tf.transpose(examples_x_activations)
    return activations_x_examples


def make_model_by_parameters(filename):
    model = tf.keras.list_models.load_model(filename)
    return model


def _examples_x_activations_for_input(x, model, num_skipped_layers_from_start):
    first_layer = True
    skipped_iterations = 0
    for layer in model.layers:
        if skipped_iterations < num_skipped_layers_from_start:
            x = layer(x)
            skipped_iterations += 1
        else:
            x = layer(x)
            examples_x_neurons = np.reshape(np.copy(x.numpy()), newshape=(x.shape[0], -1))
            if first_layer:
                activations_bd = MappedMatrix(array=examples_x_neurons)
                first_layer = False
            else:
                activations_bd.concatenate(examples_x_neurons)
    return activations_bd


def _examples_x_activations_for_input_tf(x, model, num_skipped_layers_from_start):
    first_layer = True
    skipped_iterations = 0
    for layer in model.layers:
        if skipped_iterations < num_skipped_layers_from_start:
            x = layer(x)
            skipped_iterations += 1
        else:
            x = layer(x)
            examples_x_neurons = tf.reshape(x, (tf.shape(x).numpy()[0], -1))
            if first_layer:
                activations_bd = examples_x_neurons
                first_layer = False
            else:
                activations_bd = tf.concat([activations_bd, examples_x_neurons], axis=1)

    return activations_bd
