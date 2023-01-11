

import numpy as np
from numpy.random import multivariate_normal


class Particles:
    def __init__(self, coods):
        self.coods = coods


class CoordinateGenerator:

    def generate_3D_gaussian(N, mean=np.zeros(3), cov=None):

        if not cov:
            cov = np.zeros((3, 3))
            np.fill_diagonal(cov, 1.)

        """ Generate a random collection of particle coordinates assuming a 3D gaussian """

        return multivariate_normal(mean, cov, N)

    # def generate_2D_Sersic(N):
    #
    #     """ Generate a random collection of particle coordinates assuming a 2D Sersic profile """
    #
    #     return
