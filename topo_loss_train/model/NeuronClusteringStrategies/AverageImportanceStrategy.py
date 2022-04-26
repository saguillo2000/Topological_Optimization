import numpy as np
import tensorflow as tf

from model.MappedMatrix import MappedMatrix


def average_importance_clustering(activations_x_examples, number_of_neurons):
    averages = np.mean(np.abs(activations_x_examples.array), axis=1)
    neurons_idxs = list(range(activations_x_examples.array.shape[0]))
    sum_of_averages = np.sum(averages)
    probs = averages / sum_of_averages
    try:
        sampled_neurons = np.random.choice(neurons_idxs, size=number_of_neurons, replace=False, p=probs)
    except ValueError:
        sampled_neurons = np.random.choice(neurons_idxs, size=number_of_neurons, replace=True, p=probs)
    # https://github.com/open-mmlab/OpenPCDet/issues/313
    sampled_activations_x_examples = np.take(activations_x_examples.array, sampled_neurons, 0)
    return MappedMatrix(array=sampled_activations_x_examples)


def average_importance_clustering_tf(activations_x_examples, number_of_neurons):
    averages = tf.reduce_mean(tf.abs(activations_x_examples), axis=1)
    neurons_idxs = list(range(activations_x_examples.numpy().shape[0]))
    sum_of_averages = tf.math.reduce_sum(averages)
    probs = tf.math.divide(averages, sum_of_averages)

    sampled_neurons = tf.random.categorical(tf.math.log(probs), number_of_neurons)

    '''
    try:
        sampled_neurons = np.random.choice(neurons_idxs, size=number_of_neurons, replace=False, p=probs)
    except ValueError:
        sampled_neurons = np.random.choice(neurons_idxs, size=number_of_neurons, replace=True, p=probs)
    # https://github.com/open-mmlab/OpenPCDet/issues/313
    '''

    # sampled_activations_x_examples = np.take(activations_x_examples.array, sampled_neurons, 0)
    sampled_activations_x_examples = tf.gather(activations_x_examples, sampled_neurons)
    return MappedMatrix(array=sampled_activations_x_examples)
