import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True, fastmath=True)
def workIntegral(mag_values, h_ramp):
    steps = h_ramp.shape[0] - 1
    work = 0
    for i in range(steps):
        work += (h_ramp[i + 1] - h_ramp[i]) * (mag_values[i + 1] + mag_values[i]) / 2

    return work


def configPlot(ising_obj):
    ''' This modules plots the configuration of the Ising model'''
    X, Y = np.meshgrid(range(ising_obj.N), range(ising_obj.N))
    e = ising_obj.energy() / (ising_obj.N ** 2)
    m = ising_obj.magnetization() / (ising_obj.N ** 2)
    f = plt.figure(figsize=(5, 5), dpi=80)
    sp = f.add_subplot(1, 1, 1)
    plt.setp(sp.get_yticklabels(), visible=False)
    plt.setp(sp.get_xticklabels(), visible=False)
    plt.pcolormesh(X, Y, ising_obj.config, vmin=-1, vmax=1, cmap=plt.cm.RdBu)
    plt.title('t={0}, E={1}, M={2}'.format(ising_obj.time, round(e, 5), round(m, 5)))
    plt.axis('tight')
    plt.show()
