import itertools

import numpy as np


class MarkovDecisionProcess():
    """
    ...

    Args:
        transition: np.ndarray - transition matrix with dimenions (action, start state, end state)
        reward: np.ndarray - reward matrix with shape (n_states)
    """
    def __init__(self, transition, reward):
        self.transition = transition
        self.reward = reward

    @property
    def n_states(self):
        """
        Return the number of states.
        """
        assert self.reward is not None
        assert self.transition is not None
        assert self.reward.shape[0] == self.transition.shape[1]
        assert self.reward.shape[0] == self.transition.shape[2]
        return self.reward.shape[0]

    @property
    def n_actions(self):
        """
        Return the number of actions.
        """
        assert self.transition is not None
        return self.transition.shape[0]

    def value(self, policy, discount=1):
        """
        Compute the value function of the given policy.

        Args:
            policy: np.ndarray - policy vector with shape (n_states, n_actions) containing the probability of taking any action at a given state
            discount: float - discount factor

        Returns:
            np.ndarray - value function with shape (n_states)
        """
        assert self.reward is not None
        assert self.transition is not None
        n = self.n_states
        t = np.empty([n,n]) # Transition matrix induced by the policy
        for s0,s1 in itertools.product(range(n),range(n)):
            t[s0,s1] = self.transition[:,s0,s1].dot(policy[s1,:])
        i = np.eye(n)
        return np.linalg.pinv(i-discount*t) @ self.reward

    def action_value(self, policy, discount=1):
        """
        Compute the action value function of the given policy.

        Args:
            policy: np.ndarray - policy vector with shape (n_states, n_actions) containing the probability of taking any action at a given state
            discount: float - discount factor

        Returns:
            np.ndarray - action value function with shape (n_states, n_actions)
        """
        assert self.reward is not None
        assert self.transition is not None
        n = self.n_states
        m = self.n_actions
        t = np.empty([n*m,n*m]) # state-action to state-action transition matrix
        for a0,s0,a1,s1 in itertools.product(range(m),range(n),range(m),range(n)):
            t[a0*n+s0,a1*n+s1] = self.transition[a0,s0,s1] * policy[s1,a1]
        i = np.eye(n*m)
        r = np.tile(self.reward, m)
        return (np.linalg.pinv(i-discount*t) @ r).reshape(n,m)
