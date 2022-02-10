import numpy as np
import tensorflow as tf
from tensorflow import keras
import random


def _inflate(t):
    if t[0] in [0,None]:
        return (1,) + t[1:]
    return t


def _pick_indices(size: int, one_qt: int):
    return random.sample(range(size), one_qt)


def gradient_noise_01(model: tf.keras.Model,
                      dataset: tf.data.Dataset,
                      gradient_sample_qt = 10000,
                      dataset_sample_qt = 500):
    dataset.shuffle(dataset_sample_qt)

    grad_size = np.sum((np.prod(weight.shape) for weight in model.get_weights()))
    label_qt = model.output_shape[1]
    grad_indices = _pick_indices(grad_size, gradient_sample_qt)
    entropy = keras.losses.BinaryCrossentropy()

    weights_variable = model.trainable_weights
    x_dummy = tf.Variable(lambda: tf.zeros(shape=_inflate(model.input_shape)))
    y_dummy = tf.Variable(lambda: tf.zeros(shape=_inflate(model.output_shape)))

    gradients = []
    for (x, y), _ in zip(dataset, range(dataset_sample_qt)):
        x_dummy.assign(tf.expand_dims(tf.cast(x, tf.float32), 0))
        y_dummy.assign(tf.expand_dims(tf.expand_dims(tf.cast(y, tf.float32), 0), 0))

        with tf.GradientTape() as tape:
            # tape.watch([weights_variable,x_dummy,y_dummy])
            #output = model(x_dummy)
            loss_variable = entropy(model(x_dummy), y_dummy)

        grad = tape.gradient(loss_variable, weights_variable)
        del tape

        grad = tf.concat([tf.reshape(gradee,-1) for gradee in grad if gradee is not None], 0)
        grad = tf.gather(grad, grad_indices)
        gradients.append(grad)

    gradients = tf.stack(gradients).numpy()

    gradients -= np.mean(gradients, axis=0, keepdims=True)
    gradients = np.square(gradients)
    gradients = np.sum(gradients, axis=1)
    gradients = np.sqrt(gradients)
    grad_noise = np.mean(gradients)

    return grad_noise



def gradient_noise_classifier(model: tf.keras.Model,
                              dataset: tf.data.Dataset,
                              gradient_sample_qt = 10000,
                              dataset_sample_qt = 500):
    dataset.shuffle(dataset_sample_qt)

    grad_size = np.sum((np.prod(weight.shape) for weight in model.get_weights()))
    label_qt = model.output_shape[1]
    grad_indices = _pick_indices(grad_size, gradient_sample_qt)
    entropy = keras.losses.BinaryCrossentropy()

    weights_variable = model.trainable_weights
    x_dummy = tf.Variable(lambda: tf.zeros(shape=_inflate(model.input_shape)))
    y_dummy = tf.Variable(lambda: tf.zeros(shape=_inflate(model.output_shape)))

    gradients = []
    for (x, y), _ in zip(dataset, range(dataset_sample_qt)):
        x_dummy.assign(tf.expand_dims(tf.cast(x, tf.float32), 0))
        y_dummy.assign(tf.expand_dims(tf.one_hot(y, label_qt), 0))

        with tf.GradientTape() as tape:
            # tape.watch([weights_variable,x_dummy,y_dummy])
            #output = model(x_dummy)
            loss_variable = entropy(model(x_dummy), y_dummy)

        grad = tape.gradient(loss_variable, weights_variable)
        del tape

        grad = tf.concat([keras.backend.flatten(gradee) for gradee in grad], 0)
        grad = tf.gather(grad, grad_indices)
        gradients.append(grad)

    gradients = tf.stack(gradients).numpy()

    gradients -= np.mean(gradients, axis=0, keepdims=True)
    gradients = np.square(gradients)
    gradients = np.sum(gradients, axis=1)
    gradients = np.sqrt(gradients)
    grad_noise = np.mean(gradients)

    return grad_noise
