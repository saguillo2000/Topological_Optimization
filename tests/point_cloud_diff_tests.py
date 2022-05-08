import os

import imageio
from matplotlib import pyplot as plt

import gudhi as gd

from diff import *


def create_path(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def build_gif(gif_path, frames_per_image=2, extension='png'):
    # Build GIF
    filenames = list(map(lambda filename: int(filename[:-4]), os.listdir(gif_path)))
    filenames.sort()
    with imageio.get_writer(f'{gif_path}/result.gif', mode='I') as writer:
        for filename in [f'{gif_path}/{gif_filename}.{extension}' for gif_filename in filenames]:
            image = imageio.imread(filename)
            for _ in range(frames_per_image):
                writer.append_data(image)


def paint_point_cloud(point_cloud, gif_path, number, x_low, x_high, y_low, y_high):
    plt.clf()
    if number == -1:
        plt.title(f'Initial position')
    else:
        plt.title(f'Epoch {number}')
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)
    plt.plot()
    plt.savefig(f'{gif_path}/{number}.png')


def diff_point_cloud_test(point_cloud, num_epochs, lr, dim, distance: Callable[[np.array], tf.Tensor] = None):
    optimizer = tf.keras.optimizers.Adam(lr)

    gif_path = './gif'
    create_path(gif_path)

    x_low = 0.0
    x_high = 1000.0
    y_low = 0.0
    y_high = 1000.0

    losses, Dgs, Xs, grads = [], [], [], []

    for epoch in range(num_epochs + 1):
        Dg, gradients, loss = diff_point_cloud(point_cloud, dim, distance)
        print(Dg.numpy())
        gd.plot_persistence_diagram(np.array(Dg.numpy()))

        Dgs.append(Dg.numpy())
        Xs.append(point_cloud.numpy())
        losses.append(loss.numpy())

        print(compute_total_persistence(Dg.numpy()))

        grads.append(gradients[0].numpy())
        optimizer.apply_gradients(zip(gradients, [point_cloud]))
        paint_point_cloud(point_cloud, gif_path, epoch, x_low, x_high, y_low, y_high)

    build_gif(gif_path)
    return losses, Dgs, Xs, grads
