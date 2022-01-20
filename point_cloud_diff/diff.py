from gudhi.wasserstein import wasserstein_distance

from point_cloud_diff.filtrations import *


def diff_point_cloud(X, num_epochs, lr, dim):

    XTF = tf.Variable(X, tf.float32)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    losses, Dgs, Xs, grads = [], [], [], []

    for epoch in range(num_epochs + 1):
        with tf.GradientTape() as tape:
            Dg = RipsModel(X=XTF, mel=10, dim=dim, card=10).call()
            loss = -wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)

        Dgs.append(Dg.numpy())
        Xs.append(XTF.numpy())
        losses.append(loss.numpy())

        gradients = tape.gradient(loss, [XTF])
        grads.append(gradients[0].numpy())
        optimizer.apply_gradients(zip(gradients, [XTF]))
