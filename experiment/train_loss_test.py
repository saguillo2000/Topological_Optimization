from models import *
from datasets import *
from topo_loss_train.train import *


def training():
    return 0


if __name__ == '__main__':
    train_10, test_10, INPUT_SIZE_10, OUTPUT_SIZE_10 = dataset_CIFAR10(0.01)
    train_100, test_100, INPUT_SIZE_100, OUTPUT_SIZE_100 = dataset_CIFAR100(0.01)
    train_MNIST, test_MNSIT, INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST = dataset_MNIST(0.01)

    Models_10 = Models(INPUT_SIZE_10, OUTPUT_SIZE_10)
    Models_100 = Models(INPUT_SIZE_100, OUTPUT_SIZE_100)
    Models_MNIST = Models(INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST)

    optimizer = tf.keras.optimizers.Adam()

    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering_tf)


