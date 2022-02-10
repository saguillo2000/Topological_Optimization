import numpy as np

from model.Metrics.Metric import NotRepresentableException

raise NotImplementedError("This metric hasn't been adapted to use tensorflow")
raise NotImplementedError("This metric reduction/_reduction methods have not been vectorized")

class CorrelationSimilarity(Metric):
    """
    2-corr(x,y)
    Though close, this doesn't satisfy the triangle inequality, hence being described as a similarity
    """

    diameter = 2

    @staticmethod
    def _reduce(x):
        x -= np.mean(x)
        norm = np.linalg.norm(x)
        if norm<1e-10: raise NotRepresentableException()
        x /= norm
        return x

    @staticmethod
    def _distance(x, y):
        return 2 - (x @ y.T)
