import pytest

import numpy as np

import random_mdp.random.transition
import random_mdp.random.reward
import random_mdp.random.policy
from random_mdp.mdp import MarkovDecisionProcess
from random_mdp.matrix import TransitionMatrix, PolicyMatrix


@pytest.mark.parametrize("n_states, n_actions", [(2,2),(3,2),(4,5)])
def test_same_value_if_actions_swapped(n_states, n_actions):
    """ If the actions are swapped, the value of each state-action pair should still be the same. """
    permutation1 = list(range(n_actions))
    permutation2 = list(reversed(permutation1))

    transition = random_mdp.random.transition.uniform(n_states, n_actions)
    transition1 = TransitionMatrix.from_array(transition[permutation1,...])
    transition2 = TransitionMatrix.from_array(transition[permutation2,...])

    reward = random_mdp.random.reward.uniform(n_states)
    policy = random_mdp.random.policy.uniform(n_states, n_actions)
    policy1 = PolicyMatrix.from_array(policy[...,permutation1])
    policy2 = PolicyMatrix.from_array(policy[...,permutation2])

    mdp1 = MarkovDecisionProcess(transition1, reward)
    mdp2 = MarkovDecisionProcess(transition2, reward)

    q1 = mdp1.action_value(policy1,discount=0.9)
    q2 = mdp2.action_value(policy2,discount=0.9)

    assert np.allclose(q1[...,permutation2],q2)
