import numpy as np

from model.homology.CompositePersistenceDiagram import CompositePersistenceDiagram
from model.homology.PersistenceDiagram import PersistenceDiagram


class AllDims:
    def __contains__(self, _): return True


all_dims = AllDims() # cannot be static method


def is_collection(x):
    return hasattr(x,"__contains__")


def alpha_weighted_average_life(persistence_diagram: PersistenceDiagram, alpha: float, dimensions=all_dims):
    if not is_collection(dimensions): dimensions = [dimensions]
    points = filter(lambda point: point.dim in dimensions, persistence_diagram.points)

    life = lambda point: point.death - point.birth
    values = map(lambda point: life(point)*life(point)**alpha, points)

    val_without_inf = filter(lambda val: not np.isinf(val), values)
    return np.mean(np.fromiter(val_without_inf, dtype=np.double))


def alpha_weighted_average_midlife(persistence_diagram: PersistenceDiagram, alpha: float, dimensions=all_dims):
    if not is_collection(dimensions): dimensions = [dimensions]
    points = filter(lambda point: point.dim in dimensions, persistence_diagram.points)

    life = lambda point: point.death - point.birth
    midlife = lambda point: (point.death + point.birth)/2
    values = map(lambda point: midlife(point)*life(point)**alpha, points)

    val_without_inf = filter(lambda val: not np.isinf(val), values)
    return np.mean(np.fromiter(val_without_inf, dtype=np.double))


# average_life = partial(alpha_weighted_average_life, alpha=0)
# average_midlife = partial(alpha_weighted_average_midlife, alpha=0)
# weighted_average_life = partial(alpha_weighted_average_life, alpha=1)
# weighted_average_midlife = partial(alpha_weighted_average_midlife, alpha=1)


def weighted_average_life(persistence_diagram: PersistenceDiagram, dimensions=all_dims):
    return alpha_weighted_average_life(persistence_diagram, 1, dimensions=dimensions)


def weighted_average_midlife(persistence_diagram: PersistenceDiagram, dimensions=all_dims):
    return alpha_weighted_average_midlife(persistence_diagram, 1, dimensions=dimensions)


def average_life(persistence_diagram: PersistenceDiagram, dimensions=all_dims):
    if isinstance(persistence_diagram, PersistenceDiagram):
        return alpha_weighted_average_life(persistence_diagram, 0, dimensions=dimensions)
    elif isinstance(persistence_diagram, CompositePersistenceDiagram):
        if not is_collection(dimensions): dimensions = [dimensions]
        total_weight = 0
        accumulated_life = 0
        for dimension in persistence_diagram.points_of_dim.keys():
            if dimension not in dimensions: continue
            pts = persistence_diagram.points_of_dim[dimension]
            total_weight += np.sum(pts[:,2])
            accumulated_life += np.sum((pts[:,1]-pts[:,0])*pts[:,2])
        return accumulated_life/total_weight


def average_midlife(persistence_diagram: PersistenceDiagram, dimensions=all_dims):
    if isinstance(persistence_diagram, PersistenceDiagram):
        return alpha_weighted_average_midlife(persistence_diagram, 0, dimensions=dimensions)
    elif isinstance(persistence_diagram, CompositePersistenceDiagram):
        if not is_collection(dimensions): dimensions = [dimensions]
        total_weight = 0
        accumulated_midlife = 0
        for dimension in persistence_diagram.points_of_dim.keys():
            if dimension not in dimensions: continue
            pts = persistence_diagram.points_of_dim[dimension]
            total_weight += np.sum(pts[:,2])
            accumulated_midlife += np.sum((pts[:,1]+pts[:,0])*pts[:,2])/2
        return accumulated_midlife/total_weight
