import tensorflow as tf
from tensorflow.keras import datasets
from topo_loss_train.train import reduce_dataset


def dataset_MNIST(reduction=None):
    INPUT_SIZE = (28, 28)
    OUTPUT_SIZE = 10

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images / 255.0

    train_labels = tf.reshape(train_labels, [60000, 1])

    train_images, train_labels = reduce_dataset(train_images, train_labels, reduction)

    train = (train_images, train_labels)

    return train, INPUT_SIZE, OUTPUT_SIZE


def dataset_CIFAR10(reduction=None):
    INPUT_SIZE = (32, 32, 3)
    OUTPUT_SIZE = 10

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images / 255.0

    train_images, train_labels = reduce_dataset(train_images, train_labels, reduction)

    train = (train_images, train_labels)

    return train, INPUT_SIZE, OUTPUT_SIZE


def dataset_CIFAR100(reduction=None):
    INPUT_SIZE = (32, 32, 3)
    OUTPUT_SIZE = 100

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    train_images = train_images / 255.0

    train_images, train_labels = reduce_dataset(train_images, train_labels, reduction)

    train = (train_images, train_labels)

    return train, INPUT_SIZE, OUTPUT_SIZE
