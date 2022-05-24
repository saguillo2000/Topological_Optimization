import os.path
from functools import partial

from model import NeuronSpace
from models import *
from datasets import *
from experiment.train_experiment_loss import train_experiment, serialize
from point_cloud_diff_tests import create_path
from topological_descriptors import *


def training(model, train, topo_descriptor, path):
    epochs = 30
    optimizer = tf.keras.optimizers.Adam()
    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering_tf)

    train_experiment(epochs, model, optimizer, topo_descriptor,
                     train, neuron_space_strategy, path)


def train_dataset(train, input_shape, output_shape, dataset_name):
    topo_descriptors = [(compute_total_persistence, 'total_persistence'),
                        (compute_group_persistence, 'group_persistence')]
    create_path(dataset_name)

    for topo_descriptor, folder_name in topo_descriptors:
        models = Models(input_shape, output_shape)
        path = os.path.join(dataset_name, folder_name)
        create_path(folder_name)

        for model in models.list_models():
            print('HELLO')
            path = os.path.join(path, model.name)
            print(path)
            create_path(path)
            training(model, train, topo_descriptor, path)

        serialize(dataset_name + '/Models', models)


if __name__ == '__main__':
    '''
    For each dataset we are going to train 4 different models and save their results
    '''
    train_10, INPUT_SIZE_10, OUTPUT_SIZE_10 = dataset_CIFAR10(0.01)
    train_100, INPUT_SIZE_100, OUTPUT_SIZE_100 = dataset_CIFAR100(0.01)
    train_MNIST, INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST = dataset_MNIST(0.01)

    train_dataset(train_10, INPUT_SIZE_10, OUTPUT_SIZE_10, 'CIFAR10')
    train_dataset(train_100, INPUT_SIZE_100, OUTPUT_SIZE_100, 'CIFAR100')
    train_dataset(train_MNIST, INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST, 'MNSIT')
