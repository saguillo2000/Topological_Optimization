import numpy as np

from model.homology.PersistenceLandscape import PersistenceLandscape


def accepts_persistence_diagram(f):
    def landscaped_fun(*args,**kwargs):
        persistence_something = args[0]
        if not isinstance(persistence_something, PersistenceLandscape):
            persistence_something = PersistenceLandscape(persistence_something)
        return f(persistence_something, *args[1:], **kwargs)
    landscaped_fun.__name__ = f.__name__
    return landscaped_fun


@accepts_persistence_diagram
def landscape_norm(landscape: PersistenceLandscape, k_layers, p):
    norms = []
    for layer in range(k_layers):
        if layer >= len(landscape.envelopes_l_k):
            norms.append(0)
            continue
        nodes = landscape.envelopes_l_k[layer]

        integral = 0
        for left_peak, right_peak in zip(nodes,nodes[1:]):
            height_diff = np.abs(left_peak[1]-right_peak[1])
            if height_diff == 0: continue
            min_height = min(left_peak[1],right_peak[1])
            integral += (np.power(min_height+height_diff,p+1)-np.power(min_height,p+1))/(p+1)
        norm = np.power(integral, 1/p)
        norms.append(norm)
    return norms