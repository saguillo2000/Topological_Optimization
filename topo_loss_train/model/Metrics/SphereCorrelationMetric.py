from math import pi

import tensorflow as tf

from Configuration.Constants import Random as RandomConstants
from model.Metrics.CompleteManifold import CompleteManifold


class SphereCorrelationMetric(CompleteManifold):
    """
    arccos(<x,y>)
    this is a geodesic metric on the unit hypersphere manifold S_(ambient_dim - 1)
    points are projected to be on this hypersphere
     (as embedded in the perpendicular space of (1,...,1) in R^ambient_dim )
    note that for points in the sqrt(ambient dim) sphere, module projection, this is arccos(correlation(x,y))
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
        x -= tf.reduce_mean(x, axis=1, keepdims=True)
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

    def sample(self, n, seed=RandomConstants.DEFAULT_SEED):
        # recall that a sum of orthogonal normals is isotropic
        sampled = tf.random.normal((n, self.ambient_dim), mean=0.0, stddev=1.0, seed=seed)
        sampled = self.reduce(sampled)

        if len(sampled)<n:
            new_seed = tf.random.uniform(shape=[], minval=0, maxval=2**16-1, seed=seed, dtype=tf.int64)
            sampled = tf.concat([sampled,self.sample(n-len(sampled),seed=new_seed)],0)
        return sampled
