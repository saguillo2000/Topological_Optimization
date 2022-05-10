import matplotlib.pyplot as plt
import pickle
import numpy as np


def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_persistence_diagram(dgm, num_diagram):
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

    plt.savefig('CIFAR10/2_hidden/{num}.pdf'.format(num=num_diagram))


if __name__ == '__main__':
    dgms = open_pickle('CIFAR10/2_hidden/2_hiddenPersistenceDiagrams.pkl')

    num_diagram = 1
    for dgm in dgms:
        plot_persistence_diagram(dgm, num_diagram)
        num_diagram += 1
