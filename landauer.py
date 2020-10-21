import numpy as np
import matplotlib.pyplot as plt
from ising import sampleRun

# Parameters
N = 4
num_samples = 1000
h_max = .5
n_steps = 200
T = 1.5

W, _, M = sampleRun(N, h_max, n_steps, T, num_samples)

print('')
print('Average Work / Tlog(2): ', np.mean(W)/(T*np.log(2)))
print('Average Normalized Magnetization: ', M/(num_samples*N*N))
print('Jarzynski (global average): ', np.mean(np.exp(-np.array(W)/T)))

plt.hist(W, density=True, bins=500)
plt.title('Work distribution over {0} realizations'.format(num_samples))
plt.show()
