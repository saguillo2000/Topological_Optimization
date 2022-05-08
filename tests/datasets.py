from tensorflow.keras import datasets


def dataset_MNIST():
    INPUT_SIZE = (28, 28)
    OUTPUT_SIZE = 10

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train = (train_images, train_labels)
    test = (test_images, test_labels)

    return train, test, INPUT_SIZE, OUTPUT_SIZE


def dataset_CIFAR10():
    INPUT_SIZE = (32, 32, 3)
    OUTPUT_SIZE = 10

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train = (train_images, train_labels)
    test = (test_images, test_labels)

    return train, test, INPUT_SIZE, OUTPUT_SIZE


def dataset_CIFAR100():
    INPUT_SIZE = (32, 32, 3)
    OUTPUT_SIZE = 100

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train = (train_images, train_labels)
    test = (test_images, test_labels)

    return train, test, INPUT_SIZE, OUTPUT_SIZE
