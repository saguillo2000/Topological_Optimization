from gudhi.wasserstein import wasserstein_distance

from filtrations import *


def diff_point_cloud(X, num_epochs, lr, dim, distance: Callable[[np.array], tf.Tensor] = None):
    # XTF = tf.Variable(X, tf.float32)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    losses, Dgs, Xs, grads = [], [], [], []

    for epoch in range(num_epochs + 1):
        with tf.GradientTape() as tape:
            Dg = RipsModel(X=X, mel=10, dim=dim, card=10, distance=distance).call()
            loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)
            # loss = compute_total_persistence(Dg)

        Dgs.append(Dg.numpy())
        Xs.append(X.numpy())
        losses.append(loss.numpy())

        print(compute_total_persistence(Dg.numpy()))
        print(loss)

        gradients = tape.gradient(loss, [X])

        print(gradients)
        grads.append(gradients[0].numpy())
        optimizer.apply_gradients(zip(gradients, [X]))

    return losses, Dgs, Xs, grads


def compute_total_persistence(dgm):

    '''

    :param dgm:
    :return:
    '''

    return - tf.math.reduce_sum(tf.math.subtract(dgm[:, 0], dgm[:, 1]))



