import os
import pathlib

import tensorflow as tf
import numpy as np

from constraints import BATCH_SIZE

TRAIN_ROOT_FOLDER = pathlib.Path(__file__).parent.absolute()

def get_dataset(autotuned=True):
    train_ds, val_ds = _get_train_and_validation_datasets()
    test_ds = _get_test_dataset()
    if autotuned:
        return _get_autotuned_datasets(train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds


def _get_autotuned_datasets(train_ds, val_ds, test_ds):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def _get_test_dataset(batch_size=BATCH_SIZE):
    test_folder = '{}/data/{}'.format(TRAIN_ROOT_FOLDER, 'test')
    X = np.load('{}/test_dataset_X.npy'.format(test_folder))
    y = np.load('{}/test_dataset_y.npy'.format(test_folder))
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=round(X.shape[0] / 2))
    return dataset.batch(batch_size)


def _get_train_and_validation_datasets(batch_size=BATCH_SIZE, train_per=0.8):
    train_folder = '{}/data/{}'.format(TRAIN_ROOT_FOLDER, 'train')
    X = np.load('{}/train_dataset_X.npy'.format(train_folder))
    y = np.load('{}/train_dataset_y.npy'.format(train_folder))
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=round(X.shape[0]/2))
    train_size = round(train_per * X.shape[0])
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset.batch(batch_size), val_dataset.batch(batch_size)


def _get_vectors_from_folder(folder):
    vectors = []
    vectors_filenames = os.listdir(folder)
    for vector_filename in vectors_filenames:
        vector_filepath = '{}/{}'.format(folder, vector_filename)
        vectors.append(np.load(vector_filepath))
    return vectors


if __name__ == "__main__":
    get_dataset()
