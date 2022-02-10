import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt

from model.homology.CompositePersistenceLandscape import CompositePersistenceLandscape
from model.homology.PersistenceDiagram import PersistenceDiagram
from model.homology.PersistenceDiagramPoint import PersistenceDiagramPoint
from model.homology.PersistenceLandscape import PersistenceLandscape


class WeightedPersistenceDiagramPoint(PersistenceDiagramPoint):
    """
    Persistence diagram points with weights
    note that these are not used internally in composite PDs,
    they are only for clarity _outside_ CPDs
    """
    def __init__(self, dim, birth, death, weight):
        super().__init__(dim, birth, death)
        self.weight = weight
    def __str__(self):
        return "<PPD: ({p.birth},{p.death}), weight {p.weight}, dim {p.dim}>".format(p=self)


class CompositePersistenceDiagram:
    """
    Persistence diagram with (possibly negative) weights for each point.

    Internally, a dict {dim:points of dim},
    where these points are in a 3 x q array
        q being the quantity of points of that dimension
        r being birth, death, weight
    """
    def __init__(self, points_of_dim: dict):
        self.points_of_dim = points_of_dim

    @classmethod
    def init_with_dimension(cls, dimension: int, points: np.ndarray):
        return cls({dimension: points})

    @classmethod
    def from_composite_persistence_landscape(cls, dimension, cpl: CompositePersistenceLandscape):
        points = []
        for envelope in cpl.envelopes:
            slope = (envelope[1:,1]-envelope[:-1,1])/(envelope[1:,0]-envelope[:-1,0])
            convexity = np.expand_dims(slope[:-1]-slope[1:],axis=1)/2
            env_points = envelope[1:-1] @ np.asarray([[1,1],[-1,1]])
            diagonal_weights = np.expand_dims(np.where(envelope[1:-1,1]!=0,0,0.5),axis=1)
            weighted_points = np.concatenate((env_points,convexity+diagonal_weights),axis=1)
            points.append(weighted_points)
        points = np.concatenate(points,axis=0)
        return cls.init_with_dimension(dimension, points)

    @classmethod
    def from_persistence_diagram(cls, pd: PersistenceDiagram):
        points_of_dim = dict()
        for point in pd:
            if point.dim not in points_of_dim.keys():
                points_of_dim[point.dim] = []
            points_of_dim[point.dim].append((point.birth,point.death,1))
        for dim in points_of_dim.keys():
            points_of_dim[dim] = np.asarray(points_of_dim[dim])
        return cls(points_of_dim)

    def __iter__(self):
        for dimension, points in self.points_of_dim.items():
            for p in points:
                yield WeightedPersistenceDiagramPoint(dimension, *p)

    def show(self, *dims):
        if len(dims)==0:
            dims = self.points_of_dim.keys()

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle = [tuple(int(h[i:i+2], 16)/256 for i in (1, 3, 5)) for h in color_cycle]
        for dimension, points in self.points_of_dim.items():
            color = color_cycle[dimension%len(color_cycle)]
            if dimension not in dims: continue
            # this assumes weights are all -1 to 1, which might not make sense depending on what you use CPDs for.

            pos_points = points[np.where(points[:,2]>0)]
            c = np.asarray([color+(a,) for a in pos_points[:,2]])
            plt.scatter(*(pos_points.T[:2]),c=c,edgecolors=c,label=dimension)

            neg_points = points[np.where(points[:,2]<0)]
            c = np.asarray([color+(-a,) for a in neg_points[:,2]])
            plt.scatter(*(neg_points.T[:2]),color='none',edgecolors=c)

        plt.legend([mpl.patches.Patch(color=color_cycle[dimension%len(color_cycle)]) for dimension in dims],
                   ["Dimension {}".format(dimension) for dimension in dims])
        plt.show()

    def save(self, file_path):
        with open(file_path,"wb") as f:
            pickle.dump(self.points_of_dim,f)

    @classmethod
    def load(cls, file_path):
        with open(file_path,"rb") as f:
            return cls(pickle.load(f))


if __name__=="__main__":
    from random import randint, uniform
    import os

    #display
    """
    dps = dict()
    for dim in [0,1,3]:
        qt = randint(5,50)
        ps = []
        for _ in range(qt):
            birth,weight = uniform(0,20), uniform(-1,1)
            death = uniform(birth,20)
            ps.append((birth,death,weight))
        dps[dim] = np.asarray(ps)
    cps = CompositePersistenceDiagram(dps)
    cps.show()
    """

    #casting from pd
    """
    cwd = os.getcwd()
    diagrams_path = os.path.join(cwd, "../../results/CIFAR10/pds")
    diagrams = []
    for diagram_path in os.listdir(diagrams_path)[:10]:
        if diagram_path == "generalization_metrics.bin": continue
        diagrams.append(PersistenceDiagram.load(os.path.join(diagrams_path, diagram_path)))

    for diagram in diagrams:
        CompositePersistenceDiagram.from_persistence_diagram(diagram).show()
    """

    #building from CPL
    """
    cwd = os.getcwd()
    diagrams_path = os.path.join(cwd,"../../results/CIFAR10/pds")
    diagrams = []
    for diagram_path in os.listdir(diagrams_path)[:10]:
        if diagram_path == "generalization_metrics.bin": continue
        diagrams.append(PersistenceDiagram.load(os.path.join(diagrams_path,diagram_path))\
                        .get_persistence_diagram_of_dim(1))

    landscapes = list(map(PersistenceLandscape,diagrams))
    composite_landscape = CompositePersistenceLandscape.from_landscapes(landscapes, x_sampling=100)

    cpl = CompositePersistenceDiagram.from_composite_persistence_landscape(1, composite_landscape)
    cpl.show()
    """

    #diagonal correction testing
    """
    pds = [PersistenceDiagram() for _ in range(2)]
    pds[0].add_point(PersistenceDiagramPoint(1,1,11))
    pds[1].add_point(PersistenceDiagramPoint(1,2,12))
    cpl = CompositePersistenceLandscape.from_landscapes([PersistenceLandscape(pd) for pd in pds])
    cpd = CompositePersistenceDiagram.from_composite_persistence_landscape(1,cpl)
    cpd.show()
    """