from diff import *


def diff_point_cloud_test(X, num_epochs, lr, dim, distance: Callable[[np.array], tf.Tensor] = None):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    losses, Dgs, Xs, grads = [], [], [], []

    for epoch in range(num_epochs + 1):
        Dg, gradients, loss = diff_point_cloud(X, dim, distance)

        Dgs.append(Dg.numpy())
        Xs.append(X.numpy())
        losses.append(loss.numpy())

        print(compute_total_persistence(Dg.numpy()))

        grads.append(gradients[0].numpy())
        optimizer.apply_gradients(zip(gradients, [X]))

    return losses, Dgs, Xs, grads
