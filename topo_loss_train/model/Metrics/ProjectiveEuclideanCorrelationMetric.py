import numpy as np
from scipy.spatial import distance_matrix

from model.Metrics.Metric import NotRepresentableException

raise NotImplementedError("This metric hasn't been adapted to use tensorflow")
raise NotImplementedError("This metric reduction/_reduction methods have not been vectorized")


class ProjectiveEuclideanCorrelationMetric(CompleteManifold):
    """
    euclidean distance in R^ambient_dim
    original input points are projected onto the positive (first dimension) half of S_(ambient_dim-2),
     as embedded in the perpendicular space of (1,...,1)
    note that for them, the euclidean distance is equivalent to sqrt(2-correlation(x,y)),
     be x,y in their original coordinates or projected
    thus, this is in some sense the closest geodesic space to the metric space induced by 2-correlation(x,y)
    it, however, is still an embedding, and geodesics in this ambient space are NOT contained in this embedding
    thus, combination operations may fall anywhere on the ambient space
    note that the sampling method actually samples the embedded space (as it is bounded)
    """

    diameter = 1

    def __init__(self,ambient_dim):
        self.ambient_dim = ambient_dim

    @staticmethod
    def exp(p,v):
        q = p+v
        if q[0]<0: q = -q
        return q

    @classmethod
    def log(cls,x,y):
        distance = cls.distance(x,y)
        quotient_distance = cls.distance(x,-y)
        if distance > quotient_distance: y = -y
        v = y-x
        if v[0]<0: v = -v
        return v

    @staticmethod
    def _reduce(x):
        x -= np.mean(x)
        norm = np.linalg.norm(x)
        if norm<1e-10: raise NotRepresentableException()
        x /= norm
        if x[0]>0:
            x = -x
        return x

    @staticmethod
    def _distance(x,y):
        return distance_matrix(x, y, p=2)

    @classmethod
    def combine(cls, x, y, a):
        distance = cls.distance(x,y)
        quotient_distance = cls.distance(x,-y)
        if distance > quotient_distance:
            y = -y
        combination = (1-a)*x + a*y
        if combination[0]<0:
            combination = -combination
        return combination

    def sample(self, n, seed=RandomConstants.DEFAULT_SEED):
        np.random.seed(seed)
        # recall that a sum of orthogonal normals is isotropic
        sampled = np.random.normal(0, 1, (n, self.ambient_dim))
        sampled = self.reduce(sampled)

        if len(sampled)<n:
            new_seed = np.random.randint(0,2**16-1)
            sampled = np.concatenate([sampled,self.sample(n-len(sampled),seed=new_seed)],axis=0)
        return sampled
