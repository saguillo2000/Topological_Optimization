from functools import partial

import numpy as np

from diff import compute_total_persistence
from filtrations import RipsModel
from model import NeuronSpace
from tensorflow.keras import losses

from experiment.datasets import *
from experiment.models import Models

import pickle

import warnings

warnings.filterwarnings("ignore")


def train_step_topo(neuron_space_strategy, model, optimizer,
                    train_full_topo_loss, topo_descriptor, inputs):
    with tf.GradientTape() as tape:
        X = neuron_space_strategy(model, inputs)  # Inputs all batch, with labels X = inputs
        Dg = RipsModel(X=X, mel=np.inf, dim=0, card=30).call()
        topo_loss = - topo_descriptor(Dg)

    gradients_topo = tape.gradient(topo_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_topo, model.trainable_variables))

    train_full_topo_loss(topo_loss)

    return Dg


@tf.function
def _compute_predictions(inputs, model):
    return model(inputs)


def serialize(fileName, content):
    outputFile = open('{}.pkl'.format(fileName), 'wb')
    pickle.dump(content, outputFile)
    outputFile.close()


def train_experiment(epochs, model, optimizer, topo_descriptor,
                     train_dataset, neuron_space_strategy, path):
    train_full_topo_loss = tf.keras.metrics.Mean(name='train_full_topo_loss')

    inputs_train, labels_train = train_dataset
    inputs_train = tf.convert_to_tensor(inputs_train)

    loss_epochs_full_topo = []
    Dgms = []

    for epoch in range(epochs):
        print('EPOCH NUM:------------', epoch, '-----------')
        dgm = train_step_topo(neuron_space_strategy, model,
                              optimizer, train_full_topo_loss,
                              topo_descriptor,inputs_train)

        template = 'TOPO: Epoch {}, Perdida: {}'
        print(template.format(epoch + 1,
                              train_full_topo_loss.result()))

        loss_epochs_full_topo.append(train_full_topo_loss.result())
        train_full_topo_loss.reset_states()

        Dgms.append(dgm)

    # TODO make the path correcto to get inside
    serialize(path + '/LossesFullTopo', loss_epochs_full_topo)
    serialize(path + '/PersistenceDiagrams', Dgms)


if __name__ == '__main__':
    train_mnist, input_size, output_size = dataset_MNIST()

    loss_object = losses.SparseCategoricalCrossentropy()

    models = Models(input_size, output_size)

    optimizer = tf.keras.optimizers.Adam()

    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering_tf)

    print('MODEL ARCHITECTURE FOR TOPO REG AND NONE TOPO REG: ')
    print(models.two_hidden.summary())

    train_experiment(30, models.two_hidden, optimizer,
                     train_mnist, compute_total_persistence, neuron_space_strategy)
