from typing import Type

import tensorflow as tf

from model.Metrics.Metric import Metric


def maxmin_sampling_clustering(activations_x_examples : tf.Tensor,
                               number_of_neurons,
                               metric : Type[Metric]
                               ):

    qt_points,ambient_dim = activations_x_examples.shape

    picked_nodes = []

    min_dists = tf.constant(float("inf"), shape=[qt_points], dtype=tf.float32)

    for i in range(number_of_neurons):
        max_dist_node = activations_x_examples[tf.math.argmax(min_dists)]
        picked_nodes.append(max_dist_node)
        min_dists = tf.math.minimum(min_dists,
                                    tf.cast(tf.squeeze(metric.distance(max_dist_node, activations_x_examples)),
                                            tf.float32))
    return tf.stack(picked_nodes)