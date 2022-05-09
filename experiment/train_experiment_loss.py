from functools import partial

import tensorflow as tf
import numpy as np

from point_cloud_diff.diff import compute_total_persistence
from filtrations import RipsModel
from model import NeuronSpace
from tensorflow.keras import losses

from experiment.datasets import *
from experiment.models import Models

import pickle

import warnings

warnings.filterwarnings("ignore")


def train_step_topo(neuron_space_strategy, model, optimizer,
                    train_full_topo_loss, inputs):
    with tf.GradientTape() as tape:
        X = neuron_space_strategy(model, inputs)  # Inputs all batch, with labels X = inputs
        Dg = RipsModel(X=X, mel=np.inf, dim=0, card=30).call()
        topo_loss = - compute_total_persistence(Dg)

    gradients_topo = tape.gradient(topo_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_topo, model.trainable_variables))

    train_full_topo_loss(topo_loss)


@tf.function
def _compute_predictions(inputs, model):
    return model(inputs)


def serialize(fileName, content):
    outputFile = open('{}.pkl'.format(fileName), 'wb')
    pickle.dump(content, outputFile)
    outputFile.close()


def train_experiment(epochs, model, optimizer,
                     train_dataset, neuron_space_strategy):
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    train_full_topo_loss = tf.keras.metrics.Mean(name='train_full_topo_loss')
    train_full_none_topo_loss = tf.keras.metrics.Mean(name='train_full_none_topo_loss')

    train_loss_topo = tf.keras.metrics.Mean(name='train_loss')

    inputs_train, labels_train = train_dataset

    print('Shape of inputs training: ', inputs_train.shape)
    print('Shape of labels training: ', labels_train.shape)

    inputs_train = tf.convert_to_tensor(inputs_train)

    print('Shape of inputs training: ', tf.shape(inputs_train))

    loss_epochs_topo = []
    loss_epochs_full_topo = []
    loss_epochs_full_none_topo = []

    for epoch in range(epochs):

        train_step_topo(neuron_space_strategy, model,
                        optimizer, train_full_topo_loss, inputs_train)

        template = 'TOPO: Epoch {}, Perdida: {}, Perdida Topo: {}, Perdida Sin Topo: {}'
        print(template.format(epoch + 1,
                              train_loss_topo.result(),
                              train_full_topo_loss.result()))

        loss_epochs_topo.append(train_loss_topo.result())
        loss_epochs_full_topo.append(train_full_topo_loss.result())
        loss_epochs_full_none_topo.append(train_full_none_topo_loss.result())

        train_loss.reset_states()
        train_full_topo_loss.reset_states()
        train_full_none_topo_loss.reset_states()

        train_loss_topo.reset_states()

    serialize('LossesEpochsTopo', loss_epochs_topo)
    serialize('LossesFullTopo', loss_epochs_full_topo)
    serialize('LossesFullNoneTopo', loss_epochs_full_none_topo)


if __name__ == '__main__':

    train_mnist, input_size, output_size = dataset_MNIST()

    loss_object = losses.SparseCategoricalCrossentropy()

    models = Models(input_size, output_size)

    optimizer = tf.keras.optimizers.Adam()

    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering_tf)

    print('MODEL ARCHITECTURE FOR TOPO REG AND NONE TOPO REG: ')
    print(models.two_hidden.summary())

    train_experiment(30, models.two_hidden, optimizer,
                     train_mnist, neuron_space_strategy)
