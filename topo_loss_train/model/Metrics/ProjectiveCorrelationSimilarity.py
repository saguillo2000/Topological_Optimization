import numpy as np

from model.Metrics.Metric import NotRepresentableException

raise NotImplementedError("This metric hasn't been adapted to use tensorflow")
raise NotImplementedError("This metric reduction/_reduction methods have not been vectorized")


class ProjectiveCorrelationSimilarity(Metric):
    """
    1-abs(corr(x,y))
    Though close, this doesn't satisfy the triangle inequality, hence being described as a similarity
    This is what was used in the original paper
    """

    diameter = 1

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
    def _distance(x, y):
        return 1 - np.abs(x @ y.T)
