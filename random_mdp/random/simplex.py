import numpy as np


def uniform(n, rng=None) -> np.ndarray:
    """
    Sample a point in the probability simplex of dimension n.
    See https://cs.stackexchange.com/a/3229

    Args:
        n: int - dimension of the simplex
        rng: np.random.RandomState - random number generator
    """
    assert n > 0
    if rng is None:
        rng = np.random
    x = rng.rand(n+1)
    x[0] = 0
    x[1] = 1
    x.sort()
    np.subtract(x[1:], x[:-1], out=x[1:])
    return x[1:]


def one_hot(n, p=1.0, rng=None) -> np.ndarray:
    """
    Sample a one-hot vector of dimension n.

    Args:
        n: int - dimension of the one-hot vector
        p: float - probability of the `hot` event
        rng: np.random.RandomState - random number generator

    Returns:
        np.ndarray - one-hot vector with shape (n), where one element has value `p` and the rest of the probability mass is evenly split amongst the remaining elements.
    """
    if rng is None:
        rng = np.random
    x = np.ones(n) * (1-p)/(n-1)
    x[rng.randint(0, n)] = p
    return x
