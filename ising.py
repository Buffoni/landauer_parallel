import numpy as np
from numpy.random import rand
import random
from numba import int32, int64, float32
from numba.types import List
from numba.experimental import jitclass
from numba import njit, prange
from utils import workIntegral

spec = [
    ('N', int32),
    ('T', float32),
    ('J', float32),
    ('h', float32),
    ('time', int32),
    ('config', List(List(int64))),
]


@jitclass(spec)
class Ising():
    """ Simulating the 2D Ising model """

    def __init__(self, N, temp, J, h, config):
        self.N = N
        self.T = temp
        self.J = J
        self.h = h
        self.time = 0
        self.config = config

    def energy(self):
        """Energy of a given configuration"""
        energy = 0
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                S = self.config[i][j]
                nb = self.config[(i + 1) % self.N][j] + self.config[i][(j + 1) % self.N] + \
                     self.config[(i - 1) % self.N][j] + self.config[i][(j - 1) % self.N]
                energy += self.J * nb * S + self.h * S
        return energy / 4

    def magnetization(self):
        """Magnetization of a given configuration"""
        mag = 0
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                mag += self.config[i][j]
        return mag

    # monte carlo moves
    def mcmove(self):
        """ This is to execute the monte carlo moves using
        Metropolis algorithm such that detailed
        balance condition is satisified"""
        beta = 1.0 / self.T
        for i in range(self.N):
            for j in range(self.N):
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                s = self.config[a][b]
                nb = self.config[(a + 1) % self.N][b] + self.config[a][(b + 1) % self.N] + \
                     self.config[(a - 1) % self.N][b] + self.config[a][(b - 1) % self.N]
                cost = -2 * self.J * s * nb - 2 * self.h * s
                if cost < 0 or rand() < np.exp(-cost * beta):
                    s *= -1
                self.config[a][b] = s

    def evolve(self, n_iter, h_values=None):
        """ This module simulates the evolution of Ising model"""
        magnetizations = np.zeros(n_iter)
        for i in range(n_iter):
            if h_values is not None:
                self.h = h_values[i]
            self.mcmove()
            magnetizations[i] = self.magnetization()
            self.time += 1
        return magnetizations

    def thermalize(self):
        for i in range(500):
            self.mcmove()

    def reset_protocol(self, n_iter=1000, hmax=2):
        """ This module simulates the reset-to-one protocol"""
        starting_h = self.h
        # Initial state is thermalized
        self.thermalize()

        # First control ramp
        upward_ramp = np.linspace(starting_h, hmax, n_iter)
        upward_m = self.evolve(n_iter, upward_ramp)

        # Second control ramp
        downward_ramp = np.linspace(hmax, starting_h, n_iter)
        downward_m = self.evolve(n_iter, downward_ramp)

        h_ramp = np.concatenate((upward_ramp, downward_ramp))
        mag_values = np.concatenate((upward_m, downward_m))

        return mag_values, h_ramp


@njit(parallel=True)
def sampleRun(N, h_max, n_steps, T, num_samples):
    W = np.zeros(num_samples)
    mag_configs = np.zeros((num_samples, 2 * n_steps))
    M = 0
    for i in prange(num_samples):
        initial_config = [[2 * random.randint(0, 1) - 1 for _ in range(N)] for _ in range(N)]
        rm = Ising(N, T, -1, 0, initial_config)
        mag_values, h_ramp = rm.reset_protocol(n_steps, h_max)
        mag_configs[i, :] = mag_values
        W[i] = workIntegral(mag_values, h_ramp)
        M += rm.magnetization()

    return W, mag_configs, M
