import numpy as np

from random_mdp.chain import state_transition, state_action_transition
from random_mdp.matrix import TransitionMatrix, PolicyMatrix, RewardMatrix, StateValueMatrix, StateActionValueMatrix


class MarkovDecisionProcess():
    """
    ...

    Args:
        transition: np.ndarray - transition matrix with dimenions (action, start state, end state)
        reward: np.ndarray - reward matrix with shape (n_states)
    """
    def __init__(self, transition: TransitionMatrix, reward: RewardMatrix):
        self.transition = transition
        self.reward = reward

    @property
    def n_states(self) -> int:
        """
        Return the number of states.
        """
        assert self.reward is not None
        assert self.transition is not None
        assert self.reward.n_states == self.transition.n_states
        return self.reward.n_states

    @property
    def n_actions(self) -> int:
        """
        Return the number of actions.
        """
        assert self.transition is not None
        return self.transition.n_actions

    def value(self, policy: PolicyMatrix, discount: float = 0.99) -> StateValueMatrix:
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

        t = state_transition(self.transition, policy)
        i = np.eye(n)

        return StateValueMatrix.from_array(
            np.linalg.pinv(i-discount*t) @ self.reward
        )

    def action_value(self, policy: PolicyMatrix, discount=0.99) -> StateActionValueMatrix:
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

        t = state_action_transition(self.transition, policy)
        i = np.eye(n*m)
        r = np.tile(self.reward, m)

        return StateActionValueMatrix.from_array(
            (np.linalg.pinv(i-discount*t) @ r).reshape(m,n).T
        )

