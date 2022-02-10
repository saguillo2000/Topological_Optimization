import math


# Definition here:
# https://arxiv.org/pdf/1507.06217.pdf
# Variance is std^2. In the original paper they use 0.1
def normalized_symmetric_Gaussian(x, y, u_x, u_y, variance=0.1):
    inside_exp = -(((x - u_x) ** 2 + (y - u_y) ** 2) / (2 * variance))
    exp_side = math.exp(inside_exp)
    return (1 / (2 * math.pi * variance)) * exp_side
