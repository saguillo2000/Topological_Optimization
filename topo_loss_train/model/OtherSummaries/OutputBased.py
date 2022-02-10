from itertools import islice

import numpy as np
import tensorflow as tf
from tensorflow import keras


def cross_entropy_classifier(model: tf.keras.Model,
                             dataset: tf.data.Dataset,
                             batch_size=64,
                             num_samples=10):

    qt_categories = model.output_shape[1]
    entropy = keras.losses.CategoricalCrossentropy()

    batched_ds = iter(dataset.shuffle(1000).repeat(-1).batch(batch_size))

    return np.mean(np.asarray([entropy(model(x),tf.one_hot(y,qt_categories))
                               for x,y in islice(batched_ds,num_samples)]))


def cross_entropy_01(model: tf.keras.Model,
                             dataset: tf.data.Dataset,
                             batch_size=64,
                             num_samples=10):

    qt_categories = model.output_shape[1]
    entropy = keras.losses.CategoricalCrossentropy()

    batched_ds = iter(dataset.shuffle(1000).repeat(-1).batch(batch_size))

    return np.mean(np.asarray([entropy(model(x),y)
                               for x,y in islice(batched_ds,num_samples)]))


def margin_classifier(model: tf.keras.Model,
                      dataset: tf.data.Dataset,
                      batch_size=64,
                      num_samples=10):

    """
    Returns the 10th percentile of the sample-dependent margin over some random sample of the dataset
    (a low percentile serves as a more robust surrogate of the minimum)
    """

    batched_ds = iter(dataset.shuffle(1000).repeat(-1).batch(batch_size))
    margin_batches = []

    for x,y in islice(batched_ds,num_samples):
        output = model(x)
        two_best = tf.math.top_k(output,k=2)
        is_correct = y == two_best.indices[:,0]
        flipped_margins = two_best.values[:,0]-two_best.values[:,1]
        margin_batches.append(tf.where(is_correct,flipped_margins,-flipped_margins))

    # this would be best with tfp, but np is more likely to be supported
    margins = np.concatenate(margin_batches)
    return np.percentile(margins,10)


def margin_01(model: tf.keras.Model,
           dataset: tf.data.Dataset,
           batch_size=64,
           num_samples=10):

    """
    Returns the 10th percentile of the sample-dependent margin over some random sample of the dataset
    (a low percentile serves as a more robust surrogate of the minimum)

    margin is only defined for classifier models. This treats networks with a scalar last layer @ value x (0 or 1)
    as if it were a classifier with 2 categories, x and 1-x
    """

    batched_ds = iter(dataset.shuffle(1000).repeat(-1).batch(batch_size))
    margin_batches = []

    for x,y in islice(batched_ds,num_samples):
        output = model(x)
        flipped_margins = 2*output-1
        is1 = y == 1
        margin_batches.append(tf.where(is1,flipped_margins,-flipped_margins))

    # this would be best with tfp, but np is more likely to be supported
    margins = np.concatenate(margin_batches)
    return np.percentile(margins,10)