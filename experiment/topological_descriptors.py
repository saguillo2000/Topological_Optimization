import tensorflow as tf
import numpy as np

from graphics import open_pickle


def compute_total_persistence(dgm):
    """
    Computes the average of death - birth of features extracted by the persistence diagram
    :param dgm: persistence diagram of point (death, birth)
    :return: float of average
    """
    return tf.math.reduce_sum(tf.math.subtract(dgm[:, 1], dgm[:, 0]))


def compute_group_persistence(dgm):
    print(tf.norm(tf.math.subtract(dgm[:, 1], dgm[:, 0]) - tf.math.reduce_mean(dgm),
                  ord='euclidean'))
    return tf.math.reduce_sum(
        tf.math.divide(compute_total_persistence(dgm),
                       tf.norm(tf.math.subtract(dgm[:, 1], dgm[:, 0]) - tf.math.reduce_mean(dgm),
                                   ord='euclidean')))


if __name__ == '__main__':
    dgms = open_pickle('CIFAR10/2_hidden/PersistenceDiagrams.pkl')
    print('Total persistence first diagram: ', compute_total_persistence(dgms[0]))
    print('Total persistence group diagram: ', compute_group_persistence(dgms[0]))

    print('Total persistence last diagram: ', compute_total_persistence(dgms[-1]))
    print('Total persistence group diagram: ', compute_group_persistence(dgms[-1]))
