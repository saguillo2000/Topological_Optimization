from tests.models import *
from tests.datasets import *
from topo_loss_train.train import *


def training():
    return 0


if __name__ == '__main__':
    train_10, test_10, INPUT_SIZE_10, OUTPUT_SIZE_10 = dataset_CIFAR10()
    train_100, test_100, INPUT_SIZE_100, OUTPUT_SIZE_100 = dataset_CIFAR100()
    train_MNIST, test_MNSIT, INPUT_SIZE_MNIST, OUTPUT_SIZE_MNIST = dataset_MNIST()


