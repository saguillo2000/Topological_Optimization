from functools import cached_property
import itertools
from random import shuffle

import numpy as np


class NeuronDistribution:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, x: float) -> float:
        # note the normalization here is just for convinience @ the client
        # distribute_neurons has to handle normalization by itself later on
        return self.raw_evaluate(x) / self.norm

    def raw_evaluate(self, x: float) -> float:
        raise NotImplementedError()

    @classmethod
    def restricted(cls, y_val: float, *args, **kwargs):
        """
        Wrapper for the constructor that locks a particular argument in place
        to ensure the normalized distribution is y_val at 1
        """
        raise NotImplementedError()

    @classmethod
    def last_layer_restricted(cls,
                              qt_layers: int, qt_neurons: int, last_layer_size: int,
                              *args, **kwargs):
        """
        Wraps cls.restricted, computing the appropiate y value at 1 so that the
        resulting neuron distribution w/ {qt_layers} layers and {qt_neurons} neurons
        has {last_layer_size} neurons at the last layer
        """

        # this actually has some non-insignificant error - mainly pertaining to how
        # cls.restricted assumes normalizing is in L1([0,1]), but
        # as far as the final distribution is concerned we simply have a uniform space
        # over {1,...,qt_neurons}/qt_neurons. The error is directly related
        # to the difference btw these two norms of the distribution.

        return cls.restricted((qt_layers) * (last_layer_size) / qt_neurons, *args, **kwargs)

    @cached_property
    def norm(self) -> float:
        """
        L1([0,1])-norm
        """
        raise NotImplementedError()

    def distribute_neurons(self, qt_layers: int, qt_neurons: int) -> np.ndarray:
        """
        distributes the amount of neurons requested in the amount of layers requested
        following the distribution (from 0 to 1)
        """

        layer_ords = np.linspace(0, 1, qt_layers)
        discrete_distribution = np.vectorize(self)(layer_ords)
        discrete_distribution *= qt_neurons / np.sum(discrete_distribution)

        neuron_qts = discrete_distribution.astype(int)
        missing_neurons = qt_neurons - np.sum(neuron_qts)
        half_neurons = discrete_distribution - neuron_qts

        for _ in range(missing_neurons):
            index = np.argmax(half_neurons)
            half_neurons[index] = 0
            neuron_qts[index] += 1

        return neuron_qts

    @staticmethod
    def default_parameter_distr(qt_params: int) -> np.ndarray:
        return None


class Pareto(NeuronDistribution):

    def __init__(self, k, offset):
        self.k = k
        self.offset = offset

    def raw_evaluate(self, x):
        return 1 / (x + 1) ** self.k + self.offset

    @classmethod
    def restricted(cls, y_val, k):

        if k == 0:
            if y_val == 1:
                offset = 1
            else:
                raise Exception("k=0 induces a degenerate distribution - value at 1 cannot be adjusted")

        if k == 1:
            offset = (y_val * np.log(2) - 0.5) / (1 - y_val)
        else:
            offset = ((k - 1) - (y_val * (2 ** k) * (1 - 2 ** (-k + 1)))) / (
                        (y_val * (2 ** k) * (k - 1)) - ((2 ** k) * (k - 1)))
        return cls(k, offset)

    @cached_property
    def norm(self):
        if self.k == 1:
            return self.offset + np.log(2)
        else:
            return self.offset + (1 - 2 ** (1 - self.k)) / (self.k - 1)

    @staticmethod
    def default_parameter_distr(qt_params: int) -> np.ndarray:
        uh = qt_params // 2
        lh = (qt_params + 1) // 2
        return np.expand_dims(np.fromiter(itertools.chain(np.linspace(-5, -0.01, uh),
                                                          np.linspace(0.001, 5, lh)),
                                          dtype=np.float64),
                              1)


class Basic(NeuronDistribution):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def raw_evaluate(self, x):
        return 1 + x * (self.a + x * self.b)

    @classmethod
    def restricted(cls, y_val, b):
        a = ((y_val - 1) + b * (y_val / 3 - 1)) / (1 - y_val / 2)
        return cls(a, b)

    @cached_property
    def norm(self):
        return 1 + self.a / 2 + self.b / 3

    @staticmethod
    def default_parameter_distr(qt_params: int) -> np.ndarray:
        return np.expand_dims(np.linspace(-0.8, 0.8, qt_params), 1)


class Piecewise(NeuronDistribution):

    def __init__(self, v, h1, h2):
        self.v = v
        self.h1 = h1
        self.h2 = h2

    def raw_evaluate(self, x):
        if x < self.v:
            l = x / self.v
            return (1 - l) * self.h1 + l
        else:
            l = (x - self.v) / (1 - self.v)
            return (1 - l) + l * self.h2

    @classmethod
    def restricted(cls, y_val, v, h1):
        h2 = (y_val * (v * (h1 + 1) / 2 + (1 - v) / 2)) / (1 - y_val * (1 - v) / 2)
        return cls(v, h1, h2)

    @cached_property
    def norm(self):
        return self.v * (1 + self.h1) / 2 + (1 - self.v) * (1 + self.h2) / 2

    @staticmethod
    def default_parameter_distr(qt_params: int):
        param_stride = np.ceil(np.sqrt(qt_params)).astype(int)
        unculled_params = list(itertools.product(np.linspace(0.2, 0.8, param_stride),
                                                 np.linspace(1.5, 3, param_stride)))
        return np.asarray(unculled_params[:qt_params])


if __name__ == "__main__":

    from matplotlib import pyplot as plt


    def show_mlp(neuron_qts, layer_separation=1, neuron_separation=0.2):
        layer_qt = len(neuron_qts)
        positions = []
        for i, neuron_qt in enumerate(neuron_qts):
            x_pos = i * layer_separation
            positions.append(np.linspace((x_pos, -neuron_separation * (neuron_qt - 1) / 2),
                                         (x_pos, neuron_separation * (neuron_qt - 1) / 2),
                                         neuron_qt)
                             )

        for pre_layer_pos, post_layer_pos in zip(positions, positions[1:]):
            pairs = itertools.product(pre_layer_pos, post_layer_pos)
            pairs = list(pairs)
            shuffle(pairs)
            qt_pairs = len(pairs)
            for i, (pre_neuron_pos, post_neuron_pos) in enumerate(pairs):
                x1, y1 = pre_neuron_pos
                x2, y2 = post_neuron_pos
                tone = i / qt_pairs
                plt.plot((x1, x2), (y1, y2), color=(tone, tone, tone))

        neuron_rad = neuron_separation / 3
        for layer_pos in positions:
            for neuron_pos in layer_pos:
                c = plt.Circle(neuron_pos,
                               radius=neuron_rad,
                               zorder=2,
                               facecolor="white",
                               edgecolor="black")
                plt.gca().add_artist(c)

        plt.axis("equal")
        # plt.gca().set_facecolor((1,0.95,0.9))

        plt.show()


    qt_curves = 30

    pareto_distributions = [Pareto.restricted(0.5, a) for a in Pareto.default_parameter_distr(qt_curves)]
    basic_distributions = [Basic.restricted(0.5, b) for b in Basic.default_parameter_distr(qt_curves)]
    piecewise_distributions = [Piecewise.restricted(0.5, v, h1) for v, h1 in
                               Piecewise.default_parameter_distr(qt_curves)]

    xs = np.linspace(0, 1, 1000)
    for distributions in [pareto_distributions,
                          basic_distributions,
                          piecewise_distributions]:
        for f in distributions:
            ys = np.vectorize(f)(xs)
            norm = np.sum(ys) / 1000
            plt.plot(xs, ys / norm)
        plt.gca().set_ylim(0, 3)
        plt.show()

    show_mlp(Pareto.last_layer_restricted(6, 100, 10, -5).distribute_neurons(6, 100))
    show_mlp(Pareto.last_layer_restricted(6, 100, 10, 5).distribute_neurons(6, 100))

    show_mlp(Basic.last_layer_restricted(6, 100, 10, -0.8).distribute_neurons(6, 100))
    show_mlp(Basic.last_layer_restricted(6, 100, 10, 0.8).distribute_neurons(6, 100))

    show_mlp(Piecewise.last_layer_restricted(6, 100, 10, 0.2, 1.5).distribute_neurons(6, 100))
    show_mlp(Piecewise.last_layer_restricted(6, 100, 10, 0.8, 1.5).distribute_neurons(6, 100))
    show_mlp(Piecewise.last_layer_restricted(6, 100, 10, 0.2, 3).distribute_neurons(6, 100))
    show_mlp(Piecewise.last_layer_restricted(6, 100, 10, 0.8, 3).distribute_neurons(6, 100))
