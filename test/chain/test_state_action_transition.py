import numpy as np

import random_mdp.random.transition
import random_mdp.random.policy
from random_mdp.chain import state_action_transition
from random_mdp.matrix import PolicyMatrix


def test_first_action():
    n_states = 2
    n_actions = 2
    transition = random_mdp.random.transition.uniform(n_states, n_actions)
    policy = PolicyMatrix.from_array(
        np.array([
            [1,0] for _ in range(n_states)
        ])
    )
    output = state_action_transition(transition,policy)
    assert (output[:n_states,:n_states] == transition[0,:,:]).all()


def test_second_action():
    n_states = 2
    n_actions = 2
    transition = random_mdp.random.transition.uniform(n_states, n_actions)
    policy = PolicyMatrix.from_array(
         np.array([
            [0,1] for _ in range(n_states)
        ])
    )
    output = state_action_transition(transition,policy)
    assert (output[n_states:,n_states:] == transition[1,:,:]).all()


def test_uniform_policy():
    """ If the policy is uniform, the resulting state transition matrix should be the mean over the transition matrices over all actions. """
    n_states = 2
    n_actions = 2
    transition = random_mdp.random.transition.uniform(n_states, n_actions)
    policy = PolicyMatrix.from_array(
        np.array([
            [0.5,0.5] for _ in range(n_states)
        ])
    )
    output = state_action_transition(transition,policy)
    assert (output[:n_states,:n_states] == transition[0,:,:]/2).all()


def test_result_is_probability():
    """ The resulting state transition matrix must be a probability matrix. """
    n_states = 2
    n_actions = 2
    transition = random_mdp.random.transition.uniform(n_states, n_actions)
    policy = random_mdp.random.policy.uniform(n_states, n_actions)
    output = state_action_transition(transition,policy)
    assert np.allclose(output.sum(1), 1)


