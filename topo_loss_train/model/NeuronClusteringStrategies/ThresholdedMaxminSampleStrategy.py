from typing import Type

import numpy as np
import tensorflow as tf

from model.Metrics.Metric import Metric


def thresholded_maxmin_sampling_clustering_exact(activations_x_examples : tf.Tensor,
                                           number_of_neurons,
                                           metric : Type[Metric],
                                           std_threshold = -1):
    """
    maxmin, but attempts to cull extremal nodes
    the thresholding is performed with voronoi cells
        a distribution of the amount of nodes in each prototype's voronoi cell is drawn
        and prototypes whose amount falls beneath -std_threshold standard deviations of the mean are deemed extremal
    always returns at least the amount of nodes requestes (usually exactly)
    """

    qt_points,ambient_dim = activations_x_examples.shape

    current_prototype_qt = 0
    prototypes = tf.constant(0.0,shape=[0,ambient_dim],dtype=activations_x_examples.dtype
                             )

    min_dists = tf.constant(float("inf"), shape=[qt_points], dtype=tf.float32)

    while current_prototype_qt<number_of_neurons:
        picked_nodes = []
        for _ in range(6*(number_of_neurons-current_prototype_qt)):
            max_dist_node = activations_x_examples[tf.math.argmax(min_dists)]
            picked_nodes.append(max_dist_node)
            min_dists = tf.math.minimum(min_dists,
                                        tf.cast(tf.squeeze(metric.distance(max_dist_node, activations_x_examples)),
                                                tf.float32))

        prototypes = tf.concat([prototypes,tf.stack(picked_nodes)],axis=0)

        distances = metric.distance(prototypes, activations_x_examples)
        cluster_indices = tf.math.argmin(distances, axis=0)

        # represented_count = np.zeros(picked_nodes.shape[0])
        # for index in cluster_indices: represented_count[index] += 1
        # notice this line relies on the fact that every voronoi cell contains at least one node
        _, represented_count = np.unique(cluster_indices, return_counts=True)

        threshold = np.mean(represented_count)+std_threshold*np.std(represented_count)
        reduced_prototypes = prototypes[represented_count>threshold]

        current_prototype_qt, ambient_dim = reduced_prototypes.shape

    return reduced_prototypes


def thresholded_maxmin_sampling_clustering(activations_x_examples : tf.Tensor,
                                           number_of_neurons,
                                           metric : Type[Metric],
                                           std_threshold = -1):
    """
    maxmin, but attempts to cull extremal nodes
    (this culling results in number_of_neurons being an upper bound for the amount of nodes returned, rarely the actual amount
        the amount culled can be bounded w/ chebyshev's inequality (regarding variance), though not very effectively)
    the thresholding is performed with voronoi cells
        a distribution of the amount of nodes in each prototype's voronoi cell is drawn
        and prototypes whose amount falls beneath -std_threshold standard deviations of the mean are deemed extremal
    """

    qt_points,ambient_dim = activations_x_examples.shape

    picked_nodes = []

    min_dists = tf.constant(float("inf"), shape=[qt_points], dtype=tf.float32)

    for i in range(number_of_neurons):
        max_dist_node = activations_x_examples[tf.math.argmax(min_dists)]
        picked_nodes.append(max_dist_node)
        min_dists = tf.math.minimum(min_dists,
                                    tf.cast(tf.squeeze(metric.distance(max_dist_node, activations_x_examples)),
                                            tf.float32))
    picked_nodes = tf.stack(picked_nodes)

    distances = metric.distance(picked_nodes, activations_x_examples)
    cluster_indices = tf.math.argmin(distances, axis=0)

    # represented_count = np.zeros(picked_nodes.shape[0])
    # for index in cluster_indices: represented_count[index] += 1
    # notice this line relies on the fact that every voronoi cell contains at least one node
    _, represented_count = np.unique(cluster_indices, return_counts=True)

    threshold = np.mean(represented_count)+std_threshold*np.std(represented_count)
    picked_nodes = picked_nodes[represented_count>threshold]

    return picked_nodes
