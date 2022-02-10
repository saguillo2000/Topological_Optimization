from abc import abstractmethod

import tensorflow as tf

from model.Metrics.Metric import Metric


class CompleteManifold(Metric):
    """
    Relevant operations in an abstract (to-subclass) geodesically complete riemannian manifold
    Note one should most likely also override Metric methods (_distance, reduce)
    """

    @abstractmethod
    def exp(self,
                p : tf.Tensor,
                v : tf.Tensor) -> tf.Tensor:
        """
        Exponential map
        Given a point p in the manifold, maps a vector v from the tangent space at p
        to phi(||v||), where phi is a (the) unit-speed geodesic from p in the direction of v
        """
        pass

    @abstractmethod
    def log(self,
                x : tf.Tensor,
                y : tf.Tensor) -> tf.Tensor:
        """
        Logarithmic map
        Returns a vector v, inverse of y wrt the exponential map at x
         (with the minimum norm among vectors satisfying these conditions)
        """
        pass

    def combine(self,
                x : tf.Tensor,
                y : tf.Tensor,
                a : float) -> tf.Tensor:
        """
        Generalized affine combination (1-a)x+ay
        Might be overriden to provide a faster implementation
        """
        return self.exp(x,a*self.log(x,y))

    @abstractmethod
    def sample(self, qt : int, seed=None) -> tf.Tensor:
        """
        In finite-measure manifolds (with the measure induced by the metric)
         should return a sample(/ing) from a uniform distribution wrt this measure
        otherwise, whatever sampling suits the usage
        """
        pass
