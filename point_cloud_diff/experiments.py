import os

from matplotlib import pyplot as plt

from distance_strategies.correlation_strategy import *
from point_cloud_diff_tests import diff_point_cloud_test

import imageio as imageio

"""
Steps:
    - En otro proyecto creas redes neuronales aleatorias (Hecho en mi repo, puedes copiar/pegar)
    - Coges una red aleatoria de las redes que te has generado (10 minutos)
    - Optimizas sin loss topologico y con loss topologico Entrenamiento te lo puedes copiar de cualquier tuto de tensorflow. Único desafío añadir loss topológica, que es sumar algo(10 minutos)
    - Comparas accuracies en test set TNUI (10 minutos)
Repetir para todas las redes generadas (2 horas)
Hacer gráfico que compare accuracies para todas las redes
"""


def plot_points(X):
    plt.figure()
    x = X[:, 0]
    y = X[:, 1]
    plt.scatter(x, y)
    plt.show()


def create_path(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def plot_point_movement(X_opt, lr, grads):
    pts_to_move = np.argwhere(np.linalg.norm(grads[0], axis=1) != 0).ravel()
    plt.figure()
    for pt in pts_to_move:
        plt.arrow(X_opt[0][pt, 0], X_opt[0][pt, 1], -lr * grads[0][pt, 0], -lr * grads[0][pt, 1], color='blue',
                  length_includes_head=True, head_length=.05, head_width=.1, zorder=10)

    plt.scatter(X_opt[0][:, 0], X_opt[0][:, 1], c='red', s=50, alpha=.2, zorder=3)
    plt.scatter(X_opt[0][pts_to_move, 0], X_opt[0][pts_to_move, 1], c='red', s=150, marker='o', zorder=2, alpha=.7,
                label='Step i')
    plt.scatter(X_opt[1][pts_to_move, 0], X_opt[1][pts_to_move, 1], c='green', s=150, marker='o', zorder=1, alpha=.7,
                label='Step i+1')
    plt.axis('square')
    # plt.xlim([-.7,2.3])
    # plt.ylim([-.7,2.3])
    plt.legend()
    plt.show()


def prove_euclidean_distance():
    X = np.array([[0.1, 0.], [1.5, 1.5], [0., 1.6]])
    lr = 1

    X = tf.Variable(X, tf.float32)

    losses, _, X_opt, grads = diff_point_cloud_test(point_cloud=X, num_epochs=1, lr=lr, dim=0)

    print(X_opt)

    plot_point_movement(X_opt, lr, grads)


def prove_correlation():
    X = np.array([[10., 20.], [7.5, 1.5], [-10., 100.6]])
    # X = np.random.rand(6000, 50)
    print(X)
    lr = 1

    X = tf.Variable(X, tf.float32)

    losses, _, X_opt, grads = diff_point_cloud_test(point_cloud=X, num_epochs=1, lr=lr, dim=0,
                                                    distance=distance_corr_tf)

    print(X_opt)

    plot_point_movement(X_opt, lr, grads)


def prove_homology_0(num_epochs=300):
    inferior = -1
    supreme = 1000
    point_cloud = np.random.rand(1000, 2)
    point_cloud = point_cloud * supreme + inferior
    lr = 1

    plot_points(point_cloud)

    X = tf.Variable(point_cloud, tf.float32)

    losses, _, X_opt, grads = diff_point_cloud_test(point_cloud=X, num_epochs=num_epochs, lr=lr, dim=0)


def prove_homology_1(num_epochs=100):
    point_cloud = np.random.rand(60, 2)
    lr = 1

    X = tf.Variable(point_cloud, tf.float32)

    losses, _, X_opt, grads = diff_point_cloud_test(point_cloud=X, num_epochs=num_epochs, lr=lr, dim=1)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    # prove_correlation()
    # prove_euclidean_distance()
    # prove_homologies()
    prove_homology_0()
