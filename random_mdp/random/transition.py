import numpy as np

from random_mdp.random import simplex
from random_mdp.matrix import TransitionMatrix


def _sample(num_states, num_actions, rng, func) -> TransitionMatrix:
    if rng is None:
        rng = np.random
    transition = TransitionMatrix([num_actions, num_states, num_states])
    for a in range(num_actions):
        for s in range(num_states):
            transition[a, s, :] = func(rng)
    return transition


def uniform(num_states, num_actions, rng=None) -> TransitionMatrix:
    """
    Sample a Markov decision process uniformly at random.
    """
    func = lambda rng: simplex.uniform(num_states, rng=rng)
    return _sample(num_states, num_actions, rng, func)


def one_hot(num_states, num_actions, p=1.0, rng=None) -> TransitionMatrix:
    func = lambda rng: simplex.one_hot(num_states, p=p, rng=rng)
    return _sample(num_states, num_actions, rng, func)
