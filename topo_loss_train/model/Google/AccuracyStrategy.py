import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_accuracy(dataset, model):
    dataset_generator = tfds.as_numpy(dataset)
    x_list, y_list = zip(*dataset_generator)  # * unpack operator.
    x_list = list(map(lambda x_example: x_example[np.newaxis, ...], x_list))
    x = np.concatenate(x_list, axis=0)
    y = tf.convert_to_tensor(y_list)
    predictions = tf.argmax(model.predict(x), 1, output_type=tf.dtypes.int32)
    equality = tf.equal(predictions, y)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.double))
    return accuracy
