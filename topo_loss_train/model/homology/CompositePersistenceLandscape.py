import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from model.homology.PersistenceDiagram import PersistenceDiagram
from model.homology.PersistenceLandscape import PersistenceLandscape
from model.homology.PersistenceLandscape import _eval_linear_interpolation

class CompositePersistenceLandscape:

    def __init__(self, envelopes):
        self.envelopes = envelopes
        self.max_k = len(envelopes)

    @classmethod
    def from_landscape(cls, landscape, x_sampling=None):
        return cls.from_landscapes([landscape], x_sampling=x_sampling)

    @classmethod
    def from_landscapes(cls, landscapes, x_sampling=None):
        composite_envelopes = []
        qt_landscapes = len(landscapes)
        max_k = max(map(lambda x: x.max_k,landscapes))
        for k in range(max_k):
            #print(k,max_k)
            envelopes = [np.asarray(landscape.envelopes_l_k[k])
                         for landscape in landscapes
                         if k<landscape.max_k]
            qt_nonzero_landscapes = len(envelopes)
            xs,distribution = _get_distribution_of_envelopes(envelopes,x_sampling)
            composite_envelopes.append(np.stack([xs,np.mean(distribution,axis=1)*(qt_nonzero_landscapes/qt_landscapes)],axis=-1))
        return cls(composite_envelopes)

    def plot(self, *args, **kwargs):
        for envelope in self.envelopes:
            plt.plot(*envelope.T,*args,**kwargs)

    def show(self, *args, **kwargs):
        self.plot(*args, **kwargs)
        plt.show()

    @classmethod
    def compute_confidence_band(cls, landscapes, confidence=0.95, x_sampling=None, normal=False):
        """pointwise confidence band w/ normality assumption (!) / quantile band"""
        lower_envelopes = []
        upper_envelopes = []
        qt_landscapes = len(landscapes)
        max_k = max(map(lambda x: x.max_k,landscapes))
        for k in range(max_k):
            envelopes = [np.asarray(landscape.envelopes_l_k[k])
                         for landscape in landscapes
                         if k<landscape.max_k]
            qt_nonzero_envelopes = len(envelopes)
            xs,distribution = _get_distribution_of_envelopes(envelopes,x_sampling)

            #normal distr
            if normal:
                mean = np.sum(distribution,axis=1,keepdims=True)/qt_landscapes
                var = (np.linalg.norm(distribution-mean,axis=1,keepdims=True)**2
                       +mean*(qt_landscapes-qt_nonzero_envelopes))\
                      /(qt_landscapes-1)
                adj_std = np.sqrt(var/qt_landscapes)
                displ = norm.ppf((1+confidence)/2)*adj_std
                upper_envelope = np.squeeze(mean+displ)
                lower_envelope = np.squeeze(np.maximum(0,mean-displ))

            #empirical
            else:
                lower_quantile, upper_quantile = (1-confidence)/2, 0.5+confidence/2
                #adjust for the amount of zero envelopes
                lower_quantile = 1-(1-lower_quantile)*(qt_landscapes/qt_nonzero_envelopes)
                upper_quantile = 1-(1-upper_quantile)*(qt_landscapes/qt_nonzero_envelopes)
                lower_envelope = np.quantile(distribution,lower_quantile,axis=1) if lower_quantile>0\
                                    else np.zeros(len(xs))
                upper_envelope = np.quantile(distribution,upper_quantile,axis=1) if upper_quantile>0\
                                    else np.zeros(len(xs))

            lower_envelopes.append(np.stack([xs, lower_envelope],axis=-1))
            upper_envelopes.append(np.stack([xs, upper_envelope],axis=-1))
        return cls(lower_envelopes), cls(upper_envelopes)


def _get_distribution_of_envelopes(envelopes,x_sampling):
    if x_sampling is None:
        xs = np.concatenate([envelope[:,0] for envelope in envelopes])
        xs = np.unique(xs)
    else:
        min_x = min([np.min(envelope[1:,0]) for envelope in envelopes])
        max_x = max([np.max(envelope[:-1,0]) for envelope in envelopes])
        offset = (max_x-min_x)/(x_sampling-1)
        xs = np.linspace(min_x-offset,max_x+offset,num=x_sampling+2)
        xs[0] = -np.inf
        xs[-1] = np.inf
    values = np.zeros((len(xs),len(envelopes)))
    envelope_x_indices = np.zeros((len(envelopes)),dtype=int)
    for x_index,x in enumerate(xs[:-1]):
        if x_index==0:continue
        for envelope_index,envelope in enumerate(envelopes):
            while envelope[envelope_x_indices[envelope_index],0]<x:
                envelope_x_indices[envelope_index]+=1
            values[x_index,envelope_index] = _eval_linear_interpolation(envelope[envelope_x_indices[envelope_index]-1,0],
                                                                        envelope[envelope_x_indices[envelope_index],0],
                                                                        envelope[envelope_x_indices[envelope_index]-1,1],
                                                                        envelope[envelope_x_indices[envelope_index],1],
                                                                        x)
    return xs,values



if __name__=="__main__":
    import os

    cwd = os.getcwd()
    diagrams_path = os.path.join(cwd,"../../results/CIFAR10/pds")
    diagrams = []
    for diagram_path in os.listdir(diagrams_path)[:10]:
        if diagram_path == "generalization_metrics.bin": continue
        diagrams.append(PersistenceDiagram.load(os.path.join(diagrams_path,diagram_path))\
                        .get_persistence_diagram_of_dim(1))
        #with open(os.path.join(diagrams_path,diagram_path),"rb") as f:
        #    diagrams.append(pickle.load(f)["persistence_diagram"].get_persistence_diagram_of_dim(1))

    landscapes = list(map(lambda x: PersistenceLandscape(x,10),diagrams))

    #composite_landscape = CompositePersistenceLandscape.from_landscape(landscapes[0])
    #plot_composite_landscape(composite_landscape)

    #composite_landscape = CompositePersistenceLandscape.from_landscapes(landscapes, x_sampling=100)
    #composite_landscape.show()

    lower,upper = CompositePersistenceLandscape.compute_confidence_band(landscapes,confidence=0.9,normal=False)
    [cl.plot() for cl in [lower,upper]]
    plt.show()