import tensorflow as tf
from scipy.spatial import distance_matrix

from Configuration.Constants import Random as RandomConstants
from model.Metrics.CompleteManifold import CompleteManifold


class EuclideanMetric(CompleteManifold):
    """
    regular flat euclidean space
    useful for testing
    """

    diameter = float("inf")

    def __init__(self,ambient_dim):
        self.ambient_dim = ambient_dim

    @staticmethod
    def exp(p,v):
        return p+v

    @staticmethod
    def log(x,y):
        return y-x

    @staticmethod
    def _distance(x,y):
        return distance_matrix(x, y, p=2)

    @staticmethod
    def combine(x, y, a):
        return (1-a)*x + a*y

    def sample(self, n, seed=RandomConstants.DEFAULT_SEED):
        return tf.random.uniform((n, self.ambient_dim), minval=-1, maxval=1, seed=seed)
