import os

import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import norm


def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_path(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def plot_persistence_diagram(dgm, num_diagram, path=None):
    np_dgm = dgm.numpy()
    x, y = np.split(np_dgm, 2, axis=1)

    x = x.flatten()
    y = y.flatten()

    f, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(x, y)
    ax.set(xlim=(-0.01, 1), ylim=(min(y) - 0.01, max(y) + 0.01))

    diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c=".3")

    def on_change(axes):
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        diag_line.set_data(x_lims, y_lims)

    ax.callbacks.connect('xlim_changed', on_change)
    ax.callbacks.connect('ylim_changed', on_change)

    ax.set_title(f'Persistence Diagram {num_diagram}')

    plt.show()
    # plt.savefig(path + f'/PD_{num_diagram}.pdf')


def plt_density(dgm, num_diagram, path):
    np_dgm = dgm.numpy()

    x, y = np.split(np_dgm, 2, axis=1)

    x = x.flatten()
    y = y.flatten().reshape(-1, 1)

    y_plot = np.linspace(min(y) -3 , max(y) + 3)

    fig, ax = plt.subplots()

    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(y)
    log_dens = kde.score_samples(y_plot.reshape((-1, 1)))
    ax.plot(
        y_plot,
        np.exp(log_dens),
        color="navy",
        lw=2,
        linestyle="-",
        label="Gaussian Kernel "
    )

    ax.legend(loc="upper right")
    ax.plot(y, -0.005 - 0.01 * np.random.random(y.shape[0]), "+k")

    # ax.set_xlim(min(y) - 1, max(y) + 1)
    # ax.set_ylim(-0.02, 0.7)

    ax.set_title('Density Function')

    # plt.show()
    plt.savefig(path + '/DF_{num}.pdf'.format(num=num_diagram))


def get_all_graphics():
    datasets = ['CIFAR10', 'CIFAR100', 'MNSIT']
    models_res = ['10_hidden', '2_hidden', '3_hidden', '5_hidden']

    for dataset in datasets:

        for model in models_res:
            path = dataset + '/' + model
            print(path)
            diagrams = open_pickle(path + '/PersistenceDiagrams.pkl')
            create_path(path + '/PD_pdfs')
            create_path(path + '/DF_pdfs')

            num_diagram = 1
            for diagram in diagrams:
                plot_persistence_diagram(diagram, num_diagram, path + '/PD_pdfs')
                plt_density(diagram, num_diagram, path + '/DF_pdfs')
                num_diagram += 1


if __name__ == '__main__':
    """
    dgms = open_pickle('CIFAR10/2_hidden/PersistenceDiagrams.pkl')
    plt_density(dgms[0], 0, None)
    """

    get_all_graphics()
