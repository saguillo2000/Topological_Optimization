import tensorflow as tf
from tensorflow.keras import datasets

from constraints import BATCH_SIZE

if __name__ == '__main__':
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def get_dataset():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_dataset, val_dataset = _generate_train_and_val_datasets(train_images, train_labels)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
    return train_dataset, val_dataset, test_dataset


def _generate_train_and_val_datasets(train_images, train_labels, train_per=0.8):
    train_size = round(train_per * train_images.shape[0])
    full_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    return train_dataset.batch(BATCH_SIZE), val_dataset.batch(BATCH_SIZE)
