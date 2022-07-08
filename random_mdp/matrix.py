import numpy as np


class MarkovChain(np.ndarray):
    ...


class TransitionMatrix(np.ndarray):
    """ A transition matrix of shape (n_actions, n_states, n_states).
    """
    def __getitem__(self, key):
        return super().__getitem__(key)

    @property
    def n_actions(self):
        return self.shape[0]

    @property
    def n_states(self):
        return self.shape[1]

    @classmethod
    def from_array(cls, array):
        output = TransitionMatrix(array.shape)
        output[...] = array
        return output


class RewardMatrix(np.ndarray):
    """ A reward matrix of shape (n_states,).
    """
    @property
    def n_states(self):
        return self.shape[0]

    @classmethod
    def from_array(cls, array):
        output = RewardMatrix(array.shape)
        output[...] = array
        return output


class PolicyMatrix(np.ndarray):
    """ A policy matrix of shape (n_states, n_actions).
    """
    @property
    def n_states(self):
        return self.shape[0]

    @property
    def n_actions(self):
        return self.shape[1]

    @classmethod
    def from_array(cls, array):
        output = PolicyMatrix(array.shape)
        output[...] = array
        return output


class StateValueMatrix(np.ndarray):
    """ A state value matrix of shape (n_states,).
    """
    @property
    def n_states(self):
        return self.shape[0]

    @classmethod
    def from_array(cls, array):
        output = StateValueMatrix(array.shape)
        output[...] = array
        return output


class StateActionValueMatrix(np.ndarray):
    """ A state-action value matrix of shape (n_states, n_actions).
    """
    @property
    def n_states(self):
        return self.shape[0]

    @property
    def n_actions(self):
        return self.shape[1]

    @classmethod
    def from_array(cls, array):
        output = StateActionValueMatrix(array.shape)
        output[...] = array
        return output
