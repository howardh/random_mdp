import numpy as np

from random_mdp.mdp import MarkovDecisionProcess
from random_mdp.matrix import TransitionMatrix, RewardMatrix

def test_n_actions():
    transition = TransitionMatrix.from_array(
            np.array([[[1,0],[0,1]],[[1,0],[0,1]]])
    )
    reward = RewardMatrix.from_array(
            np.array([0,0])
    )
    mdp = MarkovDecisionProcess(transition, reward)
    assert mdp.n_actions == 2

def test_n_states():
    transition = TransitionMatrix.from_array(
            np.array([[[1,0],[0,1]],[[1,0],[0,1]]])
    )
    reward = RewardMatrix.from_array(
            np.array([0,0])
    )
    mdp = MarkovDecisionProcess(transition, reward)
    assert mdp.n_states == 2
