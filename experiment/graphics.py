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


def plot_persistence_diagram(dgm, num_diagram, path):
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

    ax.set_title('Persistence Diagram')

    plt.savefig(path + '/PD_{num}.pdf'.format(num=num_diagram))


def plt_density(dgm, num_diagram, path):
    np_dgm = dgm.numpy()

    x, y = np.split(np_dgm, 2, axis=1)

    x = x.flatten()
    y = y.flatten()

    true_dens = 0.3 * norm(0, 1).pdf(y) + 0.7 * norm(5, 1).pdf(y)

    fig, ax = plt.subplots()
    ax.fill(y, true_dens, fc="blue", alpha=0.2, label="input distribution")

    ax.legend(loc="upper left")
    ax.plot(y, -0.005 - 0.01 * np.random.random(y.shape[0]), "+k")

    ax.set_xlim(min(y) - 1, max(y) + 1)
    ax.set_ylim(-0.02, 0.4)

    ax.set_title('Density Function')

    plt.savefig(path + '/DF_{num}.pdf'.format(num=num_diagram))


if __name__ == '__main__':

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
