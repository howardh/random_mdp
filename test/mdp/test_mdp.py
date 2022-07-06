import numpy as np

from random_mdp.mdp import MarkovDecisionProcess

def test_no_reward():
    transition = np.array([[[1,0],[0,1]],[[1,0],[0,1]]])
    reward = np.array([0,0])
    policy = np.array([[1,0],[0,1]])
    mdp = MarkovDecisionProcess(transition, reward)
    assert (mdp.value(policy) == np.array([0,0])).all()
    assert (mdp.action_value(policy) == np.array([[0,0],[0,0]])).all()

def test_value_functions():
    transition = np.array([[[1,0],[0,1]],[[1,0],[0,1]]])
    reward = np.array([1,0])
    policy = np.array([[1,0],[0,1]])
    mdp = MarkovDecisionProcess(transition, reward)
    r = 1/(1-0.9)
    assert np.allclose(
            mdp.value(policy,discount=0.9),
            np.array([r,0])
    )
    assert np.allclose(
            mdp.action_value(policy,discount=0.9),
            np.array([[r,0],[r,0]])
    )
