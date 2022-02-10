import bisect
import math
from collections import OrderedDict
from functools import partial, reduce, cached_property
from itertools import groupby

import numpy as np


class PersistenceLandscape:
    """
    See https://arxiv.org/pdf/1501.00179.pdf
    See https://arxiv.org/pdf/1501.00179.pdf
    """

    def __init__(self, persistence_diagram, qt_envelopes=float("inf")):
        self.envelopes_l_k = _compute_envelopes(persistence_diagram, qt_envelopes)

    """See https://arxiv.org/pdf/1810.04963.pdf for notation"""

    def evaluate(self, k, t):
        assert (1 <= k)
        if k > self.max_k:
            return 0
        k_envelope = self.envelopes_l_k[k - 1]
        return _linear_interpolation(k_envelope, t)

    @property
    def max_k(self):
        return len(self.envelopes_l_k)

    @property
    def get_family(self):
        return [partial(self.evaluate, k=idx) for idx in range(len(self.persistence_diagram.points))]

    @cached_property
    def get_support_upper_bound(self):
        """Returns x such that all k-landscapes are 0 for any input>x (filtering infinities)"""
        deaths = [p.death for p in self.persistence_diagram.points if not np.isinf(p.death)]
        return max(deaths)


def _linear_interpolation(k_envelope, value):
    interval = list(map(lambda x: x[0], k_envelope))
    idx = bisect.bisect(interval, value)
    if math.isclose(k_envelope[idx][0], value) or np.isinf(k_envelope[idx][0]):
        return k_envelope[idx][1]
    else:
        if idx - 1 == 0:
            return 0
        dist = k_envelope[idx][0] - k_envelope[idx - 1][0]
        t = (value - k_envelope[idx - 1][0]) / dist
        return (1 - t) * k_envelope[idx - 1][1] + t * k_envelope[idx][1]


def _eval_linear_interpolation(xa,xb,ya,yb,x):
    d = xb-xa
    if np.isinf(d):
        #assuming x finite
        if np.isinf(xa): return yb
        if np.isinf(xb): return ya
    ca, cb = (xb-x)/d, (x-xa)/d
    return ca*ya+cb*yb


def _envelope_product(envelope_0, envelope_1):
    """
    computes the scalar product of two envelopes

    the alg mantains two intervals, (pre_0,post_0) and (pre_1,post_1),
    corresponding to linear pieces of envelope_0 and envelope_1,
    with the heights at each endpoint being (pre_h_0, post_h_0) and (pre_h_1,post_h_1)
    the indices of post_0 and post_1 are also kept (post_idx_0, post_idx_1), for incrementation

    these intervals are incremented (slid along the domains of the envelopes)
    such that they always have a non-null intersection. Moreover, these intersections for each step
        are disjoint
        their union is the intersection of the non-zero domain of both envelopes
    thus the integral of their product is the sum of integrals in these simple domains

    (lower_limit,upper_limit) is the intersection of (pre_0,post_0) and (pre_1,post_1)
    (lower_limit_i, upper_limit_i) indicates whether lower_limit or upper_limit respectively are
    relative extrema of envelope_0, envelope_1, or both. This is important, as the height at each end of
    the interval might not be necessary to compute

    (in fact, exploiting the fact that the differential of an interval is only ever 0,1, or -1, this can be made faster)
    """
    post_idx_0 = 1
    post_idx_1 = 1
    pre_0, pre_1 = envelope_0[0][0], envelope_1[0][0]
    post_0, post_1 = envelope_0[1][0], envelope_1[1][0]
    upper_limit_i = 1 if post_0 > post_1 else 0 if post_0 < post_1 else -1
    post_h_0, post_h_1 = envelope_0[1][1], envelope_1[1][1]
    pre_h_0, pre_h_1 = envelope_0[0][1], envelope_1[0][1]

    h0 = np.zeros(len(envelope_0) + len(envelope_1))
    h1 = np.zeros(len(envelope_0) + len(envelope_1))
    ls = np.zeros(len(envelope_0) + len(envelope_1))
    index = 0
    while post_idx_0 != len(envelope_0) - 1 and post_idx_1 != len(envelope_1) - 1:
        if upper_limit_i == 1:
            post_idx_1 += 1
            pre_1, pre_h_1 = post_1, post_h_1
            post_1, post_h_1 = envelope_1[post_idx_1]
        elif upper_limit_i == 0:
            post_idx_0 += 1
            pre_0, pre_h_0 = post_0, post_h_0
            post_0, post_h_0 = envelope_0[post_idx_0]
        else:
            post_idx_0 += 1
            post_idx_1 += 1
            pre_0, pre_1 = post_0, post_1
            pre_h_0, pre_h_1 = post_h_0, post_h_1
            post_0, post_h_0 = envelope_0[post_idx_0]
            post_1, post_h_1 = envelope_1[post_idx_1]

        if post_0 > post_1:
            upper_limit_i = 1
            upper_limit = post_1
        elif post_0 < post_1:
            upper_limit_i = 0
            upper_limit = post_0
        else:
            upper_limit_i = -1  # both
            upper_limit = post_0

        if upper_limit_i == 0:
            upper_h_0 = post_h_0
            upper_h_1 = _eval_linear_interpolation(pre_1, post_1, pre_h_1, post_h_1, upper_limit)
        elif upper_limit_i == 1:
            upper_h_0 = _eval_linear_interpolation(pre_0, post_0, pre_h_0, post_h_0, upper_limit)
            upper_h_1 = post_h_1
        else:
            upper_h_0 = post_h_0
            upper_h_1 = post_h_1

        ls[index] = upper_limit
        h0[index] = upper_h_0
        h1[index] = upper_h_1
        index += 1

    ls = ls[:index]
    h0 = h0[:index]
    h1 = h1[:index]
    np.nan_to_num(h0, copy=False)
    np.nan_to_num(h1, copy=False)

    diffs = ls[1:] - ls[:-1]
    lh0, uh0 = h0[:-1], h0[1:]
    lh1, uh1 = h1[:-1], h1[1:]

    return np.sum(np.multiply(diffs, sum((np.multiply(uh0, uh1) / 3,
                                          np.multiply(uh0, lh1) / 6,
                                          np.multiply(uh1, lh0) / 6,
                                          np.multiply(lh0, lh1) / 3))))

def landscape_product(landscape_0, landscape_1):
    return sum((_envelope_product(envelope_0, envelope_1)
                for envelope_0, envelope_1
                in zip(landscape_0.envelopes_l_k, landscape_1.envelopes_l_k)))

def _compute_envelopes(persistence_diagram, qt_envelopes):
    # There is a way to conserve infinite points, see https://arxiv.org/pdf/1501.00179.pdf
    # Algorithm 4
    def get_characteristic_points(bars_ordered):
        return list(map(lambda point: ((point[0] + point[1]) / 2, (point[1] - point[0]) / 2), bars_ordered))

    def birth_cp(point):
        return point[0] - point[1]

    def death_cp(point):
        return point[0] + point[1]

    pd_points = map(lambda point: (point.birth, point.death), persistence_diagram.points)
    pd_points = list(
        filter(lambda point: (not np.isinf(np.abs(point[0]))) and (not np.isinf(np.abs(point[1]))), pd_points))
    envelopes_l_k = []
    bars = _sort_ascending_descending(pd_points)
    characteristic_points = get_characteristic_points(bars)

    while characteristic_points and qt_envelopes>0:
        qt_envelopes-=1
        lambda_n = []
        lambda_n.extend([(-np.inf, 0), (birth_cp(characteristic_points[0]), 0), characteristic_points[0]])
        i = 1
        new_characteristic_points = []
        while i < len(characteristic_points):
            p = 1
            if birth_cp(characteristic_points[i]) >= birth_cp(lambda_n[-1]) and death_cp(
                    characteristic_points[i]) > death_cp(lambda_n[-1]):
                if birth_cp(characteristic_points[i]) < death_cp(lambda_n[-1]):
                    point = ((birth_cp(characteristic_points[i]) + death_cp(lambda_n[-1])) / 2,
                             (death_cp(lambda_n[-1]) - birth_cp(characteristic_points[i])) / 2)
                    lambda_n.append(point)
                    while (i + p < len(characteristic_points)) and (
                            math.isclose(birth_cp(point), birth_cp(characteristic_points[i + p]))) and (
                            death_cp(point) <= death_cp(characteristic_points[i + p])):
                        new_characteristic_points.append(characteristic_points[i + p])
                        p += 1
                    new_characteristic_points.append(point)
                    while ((i + p) < len(characteristic_points)) and (
                            birth_cp(point) <= birth_cp(characteristic_points[i + p])) and (
                            death_cp(point) >= death_cp(characteristic_points[i + p])):
                        new_characteristic_points.append(characteristic_points[i + p])
                        p += 1
                else:
                    lambda_n.extend([(death_cp(lambda_n[-1]), 0), (birth_cp(characteristic_points[i]), 0)])
                lambda_n.append(characteristic_points[i])
            else:
                new_characteristic_points.append(characteristic_points[i])
            i += p
        lambda_n.extend([(death_cp(lambda_n[-1]), 0), (np.inf, 0)])

        characteristic_points = new_characteristic_points
        lambda_n = list(OrderedDict.fromkeys(lambda_n))
        envelopes_l_k.append(lambda_n.copy())

    return envelopes_l_k


def _sort_ascending_descending(pd_points):
    ordered_pd_points = sorted(pd_points)
    groups = groupby(ordered_pd_points, lambda x: x[0])
    list_chunks = list(map(lambda chunk: list(chunk[1]), groups))
    for chunk in list_chunks:
        chunk.reverse()
    return list(reduce(lambda result, to_append: result.extend(to_append) or result, list_chunks, list()))