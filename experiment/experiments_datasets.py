from functools import partial

from model import NeuronSpace
from models import *
from datasets import *
from experiment.train_experiment_loss import train_experiment, serialize


def training(model, train, path):
    epochs = 30
    optimizer = tf.keras.optimizers.Adam()
    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering_tf)

    train_experiment(epochs, model, optimizer,
                     train, neuron_space_strategy, path)


def train_dataset(train, input_shape, output_shape, dataset_name):
    models = Models(input_shape, output_shape)

    for model in models.list_models():
        path = '/{dataset}/{name}'.format(dataset=dataset_name, name=model.name)
        training(model, train, path)

    serialize(dataset_name+'/Models', models)


if __name__ == '__main__':
    '''
    For each dataset we are going to train 4 different models and save their results
    '''
    train_10, test_10, INPUT_SIZE_10, OUTPUT_SIZE_10 = dataset_CIFAR10(0.01)
    train_100, test_100, INPUT_SIZE_100, OUTPUT_SIZE_100 = dataset_CIFAR100(0.01)
    train_MNIST, test_MNSIT, INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST = dataset_MNIST(0.01)
