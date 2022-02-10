from abc import ABC as AbstractBaseClass, abstractmethod

import numpy as np
import tensorflow as tf


class NotRepresentableException(Exception): pass

class Metric(AbstractBaseClass):

    def reduce(self, x : tf.Tensor) -> tf.Tensor:
        """
        (Vectorized) In-place transform to simplify computation of distance
        Note that if it is known all elements will be reduced, distance computation might be modified in a way
         that does not reflect how it _should_ behave _outside_ of the representative set
        Culls non-reductible vectors if no default_value is provided
        """
        return x

    @classmethod
    def distance(cls,
                 x : tf.Tensor,
                 y : tf.Tensor) -> tf.Tensor:
        """wrapper for _distance that handles some vectorization quirks. Override _distance, not this"""
        if tf.rank(x) <= 1:
            x = tf.expand_dims(x, axis=0)
        if tf.rank(y) <= 1:
            y = tf.expand_dims(y, axis=0)
        return cls._distance(x, y)

    @abstractmethod
    def _distance(self,
                  x : tf.Tensor,
                  y : tf.Tensor) -> tf.Tensor: pass

    @classmethod
    def distance_matrix(cls, x: tf.Tensor) -> tf.Tensor:
        """
        returns (full, square) distance matrix for a set of points
        note that this may be overriden to provide a faster implementation of distance matrices for some specific norm
        """
        qt_points,_ = x.shape
        distance_matrix = np.zeros((qt_points,qt_points))

        for i in range(qt_points-1):
            distance_matrix[i,i+1:] = cls.distance(x[i],x[i+1:])

        distance_matrix += distance_matrix.T
        return tf.constant(distance_matrix)

