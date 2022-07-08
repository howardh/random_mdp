import numpy as np

import random_mdp.random.transition
import random_mdp.random.reward
import random_mdp.random.policy
from random_mdp.mdp import MarkovDecisionProcess
from random_mdp.matrix import TransitionMatrix, RewardMatrix, PolicyMatrix


def test_no_reward():
    transition = TransitionMatrix.from_array(
            np.array([[[1,0],[0,1]],[[1,0],[0,1]]])
    )
    reward = RewardMatrix.from_array(np.array([0,0]))
    policy = PolicyMatrix.from_array(np.array([[1,0],[0,1]]))
    mdp = MarkovDecisionProcess(transition, reward)
    assert (mdp.value(policy) == np.array([0,0])).all()
    assert (mdp.action_value(policy) == np.array([[0,0],[0,0]])).all()


def test_value_functions():
    transition = TransitionMatrix.from_array(
            np.array([[[1,0],[0,1]],[[1,0],[0,1]]])
    )
    reward = RewardMatrix.from_array(np.array([1,0]))
    policy = PolicyMatrix.from_array(np.array([[1,0],[0,1]]))
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


def test_same_reward_everywhere():
    transition = TransitionMatrix.from_array(
            np.array([[[0.5,0.5],[0.5,0.5]],[[0.5,0.5],[0.5,0.5]]])
    )
    reward = RewardMatrix.from_array(np.array([1,1]))
    policy = PolicyMatrix.from_array(np.array([[0.5,0.5],[0.5,0.5]]))
    mdp = MarkovDecisionProcess(transition, reward)
    r = 1/(1-0.9)
    assert np.allclose(
            mdp.value(policy,discount=0.9),
            np.array([r,r])
    )
    assert np.allclose(
            mdp.action_value(policy,discount=0.9),
            np.array([[r,r],[r,r]])
    )


def test_same_value_if_actions_swapped():
    """ If the actions are swapped, the value of each state should still be the same. """
    n_states = 5
    n_actions = 2

    permutation1 = [0,1]
    permutation2 = [1,0]

    transition = random_mdp.random.transition.uniform(n_states, n_actions)
    transition1 = TransitionMatrix.from_array(transition[permutation1,...])
    transition2 = TransitionMatrix.from_array(transition[permutation2,...])

    reward = random_mdp.random.reward.uniform(n_states)
    policy = random_mdp.random.policy.uniform(n_states, n_actions)
    policy1 = PolicyMatrix.from_array(policy[...,permutation1])
    policy2 = PolicyMatrix.from_array(policy[...,permutation2])

    mdp1 = MarkovDecisionProcess(transition1, reward)
    mdp2 = MarkovDecisionProcess(transition2, reward)

    v1 = mdp1.value(policy1,discount=0.9),
    v2 = mdp2.value(policy2,discount=0.9),

    assert np.allclose(v1, v2)
