# @See https://www.lozeve.com/files/tdanetworks.pdf
# @See https://arxiv.org/abs/1507.06217

import warnings
from functools import cached_property
from itertools import product

import numpy as np

from util import single_use_method


class IntegrableKernel:
    """
    Some R2 -> R function, interpreted as a convolution, together with
        - its partial integrals over a specified domain
        - means to compute box integrals of the function over this domain
    """
    def __init__(self,
                 kernel_function,
                 x_min, x_max,
                 y_min, y_max,
                 frequency,
                 compute_on_init = True):

        self.kernel_function = kernel_function
        self.__call__ = kernel_function.__call__

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.frequency = frequency

        self.partials = None

        if compute_on_init:
            self.compute_integrals()

    def __call__(self, x, y):
        return self.kernel_function(x,y)

    @single_use_method
    def compute_integrals(self):
        """
        Precomputes all integrals over (0,x)x(0,y) for (x,y) in (x_min,x_max)x(y_min,y_max)
        """
        x_pos_steps = int(np.floor(self.x_max/self.frequency))+1
        x_neg_steps = int(np.ceil(self.x_min/self.frequency))-1
        y_pos_steps = int(np.floor(self.y_max/self.frequency))+1
        y_neg_steps = int(np.ceil(self.y_min/self.frequency))-1
        x_diam = -x_neg_steps+1+x_pos_steps
        y_diam = -y_neg_steps+1+y_pos_steps

        # grid alignment and whatnot
        hf = self.frequency/2
        x_values = np.linspace(self.x_min-hf, self.x_max-hf, num=x_diam)
        y_values = np.linspace(self.y_min-hf, self.y_max-hf, num=y_diam)

        #indices = np.stack([np.broadcast_to(np.expand_dims(x_values,1),(x_diam,y_diam)),
        #                    np.broadcast_to(np.expand_dims(x_values,0),(x_diam,y_diam))], axis=-1)
        #partials = np.vectorize(self.phi_diff_probability_distribution)(indices)
        partials = np.zeros((x_diam,y_diam))
        for (x_index, y_index), (x, y) in zip(np.ndindex(x_diam, y_diam), product(x_values, y_values)):
            partials[x_index, y_index] = self.kernel_function(x, y)

        partials *= self.frequency*self.frequency
        np.cumsum(partials, axis=0, out=partials)
        np.cumsum(partials, axis=1, out=partials)

        self.partials = np.roll(partials, (x_pos_steps, y_pos_steps), axis=(0, 1))

    def _partial_at(self, x, y):
        """
        Returns an approximation of the integral of the function over (0,x)x(0,y)
        (using linear interpolation w/ the precomputed nodes)
        """
        float_idx_x = x/self.frequency
        float_idx_y = y/self.frequency
        top_x, top_y = int(np.ceil(float_idx_x)), int(np.ceil(float_idx_y))
        bot_x, bot_y = top_x-1, top_y-1
        top_x_weight,bot_x_weight = float_idx_x-bot_x,top_x-float_idx_x
        top_y_weight,bot_y_weight = float_idx_y-bot_y,top_y-float_idx_y
        return self.partials[top_x,top_y]*top_x_weight*top_y_weight\
               + self.partials[top_x,bot_y]*top_x_weight*bot_y_weight\
               + self.partials[bot_x,top_y]*bot_x_weight*top_y_weight\
               + self.partials[bot_x,bot_y]*bot_x_weight*bot_y_weight

    def box_integral(self, x_min, x_max, y_min, y_max):
        """
        Returns an approximation of the integral of the function over (xmin,xmax)x(ymin,ymax)
        """
        return self._partial_at(x_max, y_max)\
               - self._partial_at(x_max, y_min)\
               - self._partial_at(x_min, y_max)\
               + self._partial_at(x_min, y_min)


def persistence_weight(f):
    """
    Turns a Persistence->R function into a (birth,death)->R function
    """
    def weight(x,y):
        return f(y-x)
    weight.__name__ = "persistence_as_{}".format(f.__name__)
    return weight


class PersistenceSurface:
    """
    Persistence Surface and means to compute the Persistence Image
    both as described in https://arxiv.org/abs/1507.06217

    Note that this implementation doesn't support having a distribution function
    depend on the persistence diagram point it is centered on
    """

    def __init__(self,
                 persistence_diagram,
                 phi_diff_probability_distribution,
                 f_weighting_function,
                 default_integral_freq = 0.005):
        self.persistence_diagram = persistence_diagram
        self.phi_diff_probability_distribution = phi_diff_probability_distribution
        self.f_weighting_function = f_weighting_function
        self.points = list(filter(lambda x: not (np.isinf(x.birth) or np.isinf(x.death)), persistence_diagram.points))

        self.max_birth = max([p.birth for p in self.points])
        self.min_birth = min([p.birth for p in self.points])
        self.max_death = max([p.death for p in self.points])
        self.min_death = min([p.death for p in self.points])

        self.default_integral_freq = default_integral_freq

        if isinstance(phi_diff_probability_distribution, IntegrableKernel):
            self.integrable_kernel = phi_diff_probability_distribution
        else:
            assumed_neg_x_dist = self.max_birth
            assumed_pos_x_dist = 2*(self.max_birth-self.min_birth)
            assumed_neg_y_dist = self.max_death
            assumed_pos_y_dist = 2*(self.max_death-self.min_death)

            self.integrable_kernel = IntegrableKernel(phi_diff_probability_distribution,
                                                      -assumed_neg_x_dist, assumed_pos_x_dist,
                                                      -assumed_neg_y_dist, assumed_pos_y_dist,
                                                      self.default_integral_freq,
                                                      compute_on_init = False)

    @cached_property
    def point_weights(self):
        return [self.f_weighting_function(point.birth, point.death)
                for point in self.points]

    def get_image(self, x_min, x_max, y_min, y_max,
                  pixel_width) -> np.ndarray:
        """
        Computes and returns persistence image, as described in https://arxiv.org/abs/1507.06217
        """

        neg_x_dist, neg_y_dist = x_min-self.max_birth,y_min-self.max_death
        pos_x_dist, pos_y_dist = x_max-self.min_birth,y_max-self.min_death
        if neg_x_dist<self.integrable_kernel.x_min or pos_x_dist>self.integrable_kernel.x_max\
           or neg_y_dist<self.integrable_kernel.y_min or pos_y_dist>self.integrable_kernel.y_max:
            if self.integrable_kernel.partials is not None:
                warnings.warn("\nIntegrable kernel in persistence surface isn't large enough\n\
                (This means the kernel integrals, which had already been computed, will be recomputed.\n\
                 This extra cost can be avoided by supplying a larger kernel to the Persistence Surface constructor)")
            phi = self.integrable_kernel.kernel_function
            del self.integrable_kernel
            self.integrable_kernel = IntegrableKernel(phi, neg_x_dist, pos_x_dist, neg_y_dist, pos_y_dist,
                                                      self.default_integral_freq,
                                                      compute_on_init=False)

        self.integrable_kernel.compute_integrals()

        box_phi_integral = self.integrable_kernel.box_integral

        def box_surface_integral(xmn, xmx, ymn, ymx):
            return sum((weight * box_phi_integral(xmn-point.birth,
                                                  xmx-point.birth,
                                                  ymn-point.death,
                                                  ymx-point.death)
                        for point, weight in zip(self.points, self.point_weights)))

        image = np.zeros((pixel_width,pixel_width))
        xs = list(np.linspace(x_min,x_max,pixel_width+1))
        ys = list(np.linspace(y_min,y_max,pixel_width+1))
        for x_index,y_index in np.ndindex(pixel_width,pixel_width):
            image[x_index,y_index] = box_surface_integral(xs[x_index],xs[x_index+1],
                                                          ys[y_index],ys[y_index+1])

        return image

    def __call__(self, x, y):
        """
        Returns height of the surface at the provided (birth, death) coordinates
        """
        return sum((weight * self.integrable_kernel(x-point.birth,
                                                    y-point.death)
                    for point, weight in zip(self.points,self.point_weights)))

    def get_drawing(self, x_min, x_max, y_min, y_max,
                    pixel_width) -> np.ndarray:
        """
        Computes and returns a basic (zeroth-order) approximation of the persistance image
        this approximation works best when the underlying kernel has high variance
        (as this ensures the derivative of the surface is somewhat smaller,
         and hence evaluation is closer to integration over a small region)

        (This method is only about 4 times faster than PI computation
         the only real benefit is it doesn't necessitate computation of kernel integrals)
        """

        image = np.zeros((pixel_width,pixel_width))
        xs = list(np.linspace(x_min,x_max,pixel_width))
        ys = list(np.linspace(y_min,y_max,pixel_width))
        for x_index,y_index in np.ndindex(pixel_width,pixel_width):
            image[x_index,y_index] = self(xs[x_index],ys[y_index])

        return image


if __name__=="__main__":
    import os
    import pickle
    from matplotlib import pyplot as plt

    def get_gaussian(variance):
        def gaussian(x, y):
            return (1 / (2 * np.pi * variance)) * np.exp(-((x*x+y*y) / (2 * variance)))
        return gaussian

    cwd = os.getcwd()
    diagrams_path = os.path.join(cwd,"../results/full_diagrams/task1")
    diagram_path = os.listdir(diagrams_path)[8]
    with open(os.path.join(diagrams_path,diagram_path),"rb") as f:
        some_diagram = pickle.load(f)

    #for point in some_diagram.points:
    #    point.birth = np.random.uniform(0,1)
    #    point.death = np.random.uniform(1,1.5)

    #the persistence surface constructor requires
    # a persistence diagram
    # a distribution (an R2 -> R function), as described in the paper, independent of u
    # a weighting function

    #for instance:
    # persistence diagram
    diagram = some_diagram
    # distribution (defined above)
    distr_fun = get_gaussian(0.05)
    # weighting function
    # (persistence_weight) simply maps an R->R function to a (birth,death)->R function that depends solely on life
    weighting_func = persistence_weight(lambda x:np.clip(0,0.5,x))

    # so then
    surface = PersistenceSurface(diagram,distr_fun,weighting_func)

    # inf/sup limits for x, inf/sup limits for y, width in pixels
    im = surface.get_image(0, 2, 0, 2, 5)
    im *= 250/np.max(im)
    im = np.rot90(im)
    #Image.fromarray(im).rotate(90).show()
    plt.imshow(im)
    plt.show()

    # get_drawing returns a crude approximation of the image, using the same arguments
    plt.imshow(np.rot90(surface.get_drawing(0,2,0,2,5)))
    plt.show()

    # plot to ensure the drawing coincides with the diagram
    plt.imshow(np.flip(im,axis=0))
    plt.scatter(*zip(*[(p.birth*5/2-0.5,p.death*5/2-0.5) for p in surface.points]))
    plt.xlim(0,2)
    plt.ylim(0,2)
    plt.show()

    # internally, the persistence surface precomputes a grid of partial integrals of the distribution given
    # this has some implications. Namely, if one intends to produce:
    #   an image that doesn't encompass the entire diagram,
    #   several images that do
    #   several images of different persistence diagrams with the same underlying distribution
    # it is best to supply an IntegrableKernel object to the constructor, instead of a "naked" distribution
    # the constructor is as follows:

    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    integral_sampling_frequency = 0.05

    kernel = IntegrableKernel(distr_fun,
                              x_min,x_max,
                              y_min,y_max,
                              integral_sampling_frequency)

    # distr_fun and integral_sampling_frequency are self-explanatory
    # the limits bound a domain where integration can take place
    # in persistence surfaces, these should be
    # x_min = (smallest x coordinate in the desired image) - (largest  birth value among points in the diagram)
    # y_min = (smallest y coordinate in the desired image) - (largest  death value among points in the diagram)
    # x_max = (largest  x coordinate in the desired image) - (smallest birth value among points in the diagram)
    # y_max = (largest  y coordinate in the desired image) - (smallest death value among points in the diagram)
    # or lower/upper bounds thereof,
    # (taken over the set of persistence diagrams or images we intend to use the kernel for)

    surface = PersistenceSurface(diagram,kernel,weighting_func)
