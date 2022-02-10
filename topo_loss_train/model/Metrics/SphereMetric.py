from math import pi

import numpy as np
import tensorflow as tf

from Configuration.Constants import Random as RandomConstants
from model.Metrics.CompleteManifold import CompleteManifold


class SphereMetric(CompleteManifold):
    """
    arccos(<x,y>)
    this is a geodesic metric on the unit hypersphere manifold S_(ambient_dim - 1)
    points are projected to be on this hypersphere
    """

    diameter = 2*pi

    def __init__(self,ambient_dim):
        self.ambient_dim = ambient_dim

    @staticmethod
    def exp(p,v):
        v_norm = tf.norm(v)
        if v_norm<1e-10:
            return p
        v /= v_norm
        return p*tf.math.cos(v_norm) + v*tf.math.sin(v_norm)

    @classmethod
    def log(cls,x,y):
        y_dir = y - x*tf.tensordot(x,y)
        return y_dir*(cls.distance(x,y)/tf.norm(y_dir))

    @staticmethod
    def reduce(x):
        norms = tf.norm(x, axis=1, keepdims=True)
        if tf.reduce_min(norms)<1e-10:
            is_reducible = tf.math.greater(tf.squeeze(norms), 1e-10)
            if not tf.reduce_any(is_reducible):
                _,ambient_dim = x.shape
                return tf.constant(0.0,shape=(0,ambient_dim))
            x = tf.boolean_mask(x, is_reducible)
            norms = tf.boolean_mask(norms,is_reducible)
        x /= norms
        return x

    @staticmethod
    def _distance(x,y):
        return tf.math.acos(tf.clip_by_value(tf.matmul(x,tf.transpose(y)),-1,1))

    @classmethod
    def combine(cls, x, y, a):
        y_hat = cls.reduce(y - x * tf.tensordot(x, y))
        a_hat = a * cls.distance(x,y)
        return x * tf.math.cos(a_hat) + y_hat * tf.sin(a_hat)

    def tangent_basis(self,x : np.ndarray) -> np.ndarray:
        """returns an orthonormal basis of the tangent space at x"""
        basis = np.random.uniform(1,2,(self.ambient_dim-1,self.ambient_dim))

        # note that gram-schmidt orthonormalization is somewhat numerically unstable
        for i in range(self.ambient_dim-1):
            basis[i] -= x*np.dot(x,basis[i])
            for j in range(i):
                basis[i] -= basis[j]*np.dot(basis[j],basis[i])

        return basis.T

    def sample(self, n, seed=RandomConstants.DEFAULT_SEED):
        # recall that a sum of orthogonal normals is isotropic
        sampled = tf.random.normal((n, self.ambient_dim), mean=0.0, stddev=1.0, seed=seed)
        sampled = self.reduce(sampled)

        if len(sampled)<n:
            new_seed = tf.random.uniform(shape=[], minval=0, maxval=2**16-1, seed=seed, dtype=tf.int64)
            sampled = tf.concat([sampled,self.sample(n-len(sampled),seed=new_seed)],0)
        return sampled
