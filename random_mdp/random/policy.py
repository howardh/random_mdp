import numpy as np

from random_mdp.random import simplex
from random_mdp.matrix import PolicyMatrix


def _sample(num_states, num_actions, rng, func) -> PolicyMatrix:
    if rng is None:
        rng = np.random
    policy = PolicyMatrix([num_states, num_actions])
    for s in range(num_states):
        policy[s, :] = func(rng)
    return policy


def uniform(num_states, num_actions, rng=None) -> PolicyMatrix: 
    """
    Sample a Markov decision process uniformly at random.
    """
    func = lambda rng: simplex.uniform(num_actions, rng=rng)
    return _sample(num_states, num_actions, rng, func)


def one_hot(num_states, num_actions, p=1.0, rng=None) -> PolicyMatrix:
    func = lambda rng: simplex.one_hot(num_actions, p=p, rng=rng)
    return _sample(num_states, num_actions, rng, func)
