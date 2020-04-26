import numpy as np
# source of equations: https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html

def uniform(n, order):
    return np.arange(0, n + order + 1) / (n + order)

def open_uniform(n, order, end_divergence=1e-10):
    t = np.zeros((n + order + 1))

    if end_divergence == 0 or order == 1:
        t[-order:] = 1 + end_divergence
    else:
        t[:order] = np.arange(-end_divergence * order, 0, end_divergence)
        t[-order:] = np.arange(1, 1 + (order - 1) * end_divergence, end_divergence) + end_divergence

    t[order:-order] = np.arange(0, n - order + 1) / (n - order)
    return t

if __name__ == '__main__':
    knot_uniform = uniform(5,2)
    knot_open = open_uniform(5,2)

    print('uniform:', knot_uniform)
    print('uniform_open:', knot_open)
