import numpy as np
from ising import sampleRun
import pickle
import time
from config import *

# Parameters
n_range = config['n_range']
num_samples = config['n_samples']
h_max = config['h_max']
n_steps = config['n_steps']
T = config['temp']

for N in n_range:
    start = time.time()
    W, mag_trajectories, M = sampleRun(N, h_max, n_steps, T, num_samples)
    filename = 'samples_' + str(N) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump([W, mag_trajectories, M], f)
    end = time.time()
    print('Sampled ', str(num_samples), 'points at dimension ', str(N), '. Elapsed time: ', str(end - start))
