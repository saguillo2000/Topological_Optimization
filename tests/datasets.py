from tensorflow.keras import datasets


def dataset_MNIST():
    PROBLEM_INPUT_SIZE = (28, 28)
    PROBLEM_OUTPUT_SIZE = 10

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return 0


def dataset_CIFAR10():
    PROBLEM_INPUT_SIZE = (32, 32, 3)
    PROBLEM_OUTPUT_SIZE = 10

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return 0


def dataset_CIFAR100():
    PROBLEM_INPUT_SIZE = (32, 32, 3)
    PROBLEM_OUTPUT_SIZE = 100

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return 0
