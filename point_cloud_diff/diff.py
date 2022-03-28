from gudhi.wasserstein import wasserstein_distance

from filtrations import *


def diff_point_cloud(X, dim, distance):
    with tf.GradientTape() as tape:
        Dg = RipsModel(X=X, mel=10, dim=dim, card=10, distance=distance).call()
        # loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)
        loss = compute_total_persistence(Dg)

    gradients = tape.gradient(loss, [X])

    return Dg, gradients, loss


def compute_total_persistence(dgm):
    """
    Computes the average of death - birth of features extracted by the persistence diagram
    :param dgm: persistence diagram of point (death, birth)
    :return: float of average
    """

    return tf.math.reduce_sum(tf.math.subtract(dgm[:, 1], dgm[:, 0]))
