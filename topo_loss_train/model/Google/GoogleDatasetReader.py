import glob
import os

import tensorflow as tf


def load_google_dataset(dataset_location):
    absolute_dataset_path = os.path.abspath(dataset_location)
    train_dataset = _load_train_data(absolute_dataset_path)
    test_dataset = _load_test_data(absolute_dataset_path)
    return train_dataset, test_dataset


def load_google_train_dataset(dataset_location):
    absolute_dataset_path = os.path.abspath(dataset_location)
    train_dataset = _load_train_data(absolute_dataset_path)
    return train_dataset


def _load_test_data(dataset_location):
    return _load_data(os.path.join(dataset_location, 'test'))


def _load_train_data(dataset_location):
    return _load_data(os.path.join(dataset_location, 'train'))


def _load_data(dataset_location):
    path_to_shards = glob.glob(os.path.join(dataset_location, 'shard_*.tfrecord'))
    dataset = tf.data.TFRecordDataset(path_to_shards)
    return dataset.map(_deserialize_example)


def _deserialize_example(serialized_example):
    record = tf.io.parse_single_example(
        serialized_example,
        features={
            'inputs': tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string)
        })
    inputs = tf.io.parse_tensor(record['inputs'], out_type=tf.float32)
    output = tf.io.parse_tensor(record['output'], out_type=tf.int32)
    return inputs, output
