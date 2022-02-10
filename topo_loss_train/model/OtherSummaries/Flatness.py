import numpy as np
import tensorflow as tf
from math import sqrt

from typing import Callable


def _binary_search(bot: float, top: float,
                   fun: Callable[[float], float], target: float,
                   depth=20, epsilon_x=1e-5, epsilon_y=1e-2):
    # notice initial depth value and epsilon_x both cap the same thing
    mid = (top + bot) / 2

    if top - bot < epsilon_x: return mid
    y = fun(mid)

    if abs(y - target) < epsilon_y: return mid

    if y > target:
        return _binary_search(bot, mid, fun, target, depth=depth - 1)
    else:
        return _binary_search(mid, top, fun, target, depth=depth - 1)


@tf.function
def _predict(x, model):
    logits = model(x)
    pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    return pred


def _estimate_accuracy(model, dataset, sample_qt):
    acc = 0.0
    for _ in range(sample_qt):
        x, y = dataset.next()
        pred = _predict(x, model).numpy()
        acc += np.mean(pred == y)
    return acc / sample_qt


def _get_perturbed_weights(weights, modulus: float, magnitude_aware: bool):
    return [weight + tf.random.normal(weight.shape,
                                      stddev=modulus * tf.abs(weight) if magnitude_aware else modulus)
            for weight in weights]


def average_sensitivity(model, trained_weights, dataset, perturbation_modulus, sample_qt=20, magnitude_aware=False):
    total = 0
    for _ in range(sample_qt):
        dataset.next()
        model.set_weights(_get_perturbed_weights(trained_weights, perturbation_modulus, magnitude_aware))
        total += _estimate_accuracy(model, dataset, 10)
    return 1 - total / sample_qt


def normalize_weight_perturbation(orig_weights, perturbed_weights, max_modulus: float, magnitude_aware: bool):
    dim = sum((weight.size for weight in orig_weights))
    max_2_modulus = max_modulus * sqrt(dim)

    delta_weights = [orig_weight - perturbed_weight
                     for orig_weight, perturbed_weight in zip(orig_weights, perturbed_weights)]
    # flattened 2-norm
    if magnitude_aware:
        delta_modulus = tf.sqrt(sum((tf.reduce_sum(tf.square(delta_weight))
                                     for delta_weight in delta_weights)))
    else:
        delta_modulus = tf.sqrt(sum((tf.reduce_sum(tf.square(tf.divide(delta_weight, orig_weight)))
                                     for delta_weight, orig_weight in zip(delta_weights, orig_weights))))

        # it is alright to stop early if delta_modulus<max_2_modulus
    # since, in theory
    # (i.e. assuming worst_case_sensitivity can reliably find the error maximizer in the sphere)
    # this ensures that the perturbation_modulus -> error function is nondecreasing
    # (though in the relevant worst-case scenario, it becomes flat after a point)
    # this avoids potential screw-ups in the binary search
    if delta_modulus < max_2_modulus:
        return perturbed_weights

    correction_factor = max_2_modulus / delta_modulus
    corrected_weights = []  # gc hell
    for orig_weight, delta_weight in zip(orig_weights, delta_weights):
        corrected_weights.append(orig_weight + correction_factor * delta_weight)
    return corrected_weights


def worst_case_sensitivity(model, trained_weights, dataset, perturbation_modulus,
                           sample_qt=3, steps_qt=10, magnitude_aware=False):

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    @tf.function
    def deoptimize(data):
        """Makes model as bad as possible"""
        x, y = data
        y = tf.one_hot(y, 10)
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = tf.math.negative(loss_fn(logits, y))
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    min_accuracy = 1
    for _ in range(sample_qt):
        model.set_weights(_get_perturbed_weights(trained_weights, perturbation_modulus, magnitude_aware))
        for _ in range(steps_qt):
            data_pair = dataset.next()
            deoptimize(data_pair)
            new_weights = model.get_weights()
            new_weights = normalize_weight_perturbation(trained_weights, new_weights, perturbation_modulus,
                                                        magnitude_aware)
            model.set_weights(new_weights)
        min_accuracy = min(min_accuracy, _estimate_accuracy(model, dataset, 10))

        # this only makes sense when a binary search for 0.1 is being performed
        if min_accuracy < 0.89: break
    return 1 - min_accuracy


# TODO: figure out if there's a way to optimize this
def pacbayes(model: tf.keras.Model,
             dataset: tf.data.Dataset,
             magnitude_aware=True):

    trained_weights = model.get_weights()
    batched_ds = iter(dataset.shuffle(1000).repeat(-1).batch(64))

    score = _binary_search(0, 1,
                           lambda x: average_sensitivity(model, trained_weights, batched_ds, x,
                                                         magnitude_aware=magnitude_aware),
                           0.1)
    return 1/(score*score)


def sharpness(model: tf.keras.Model,
              dataset: tf.data.Dataset,
              magnitude_aware=True):

    trained_weights = model.get_weights()
    batched_ds = iter(dataset.shuffle(1000).repeat(-1).batch(64))

    score = _binary_search(0, 0.05,
                           lambda x: worst_case_sensitivity(model, trained_weights, batched_ds, x,
                                                            magnitude_aware=magnitude_aware),
                           0.1)
    return 1 / (score * score)