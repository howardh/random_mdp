import numpy as np


def uniform(num_states, r_min=0.0, r_max=1.0, rng=None):
    if rng is None:
        rng = np.random
    x = rng.uniform(r_min, r_max, [num_states])
    return x


def one_hot(num_states, r_min=0.0, r_max=1.0, rng=None):
    """
    Sample a one-hot vector of dimension num_states.
    """
    if rng is None:
        rng = np.random
    x = np.ones(num_states) * r_min
    x[rng.randint(0, num_states)] = r_max
    return x


def normal(num_states, r_mean=0.0, r_std=1.0, rng=None):
    if rng is None:
        rng = np.random
    x = rng.normal(r_mean, r_std, [num_states])
    return x
