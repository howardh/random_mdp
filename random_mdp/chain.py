""" Functions for creating Markov chains from a transition matrix and a policy. """

import itertools

from random_mdp.matrix import MarkovChain, TransitionMatrix, PolicyMatrix


def state_transition(transition: TransitionMatrix, policy: PolicyMatrix):
    n = transition.shape[1]
    t = MarkovChain([n,n])
    for s0,s1 in itertools.product(range(n),range(n)):
        t[s0,s1] = transition[:,s0,s1].dot(policy[s0,:])
    return t


def state_action_transition(transition: TransitionMatrix, policy: PolicyMatrix):
    n = transition.shape[1] # number of states
    m = transition.shape[0] # number of actions
    t = MarkovChain([n*m,n*m]) # state-action to state-action transition matrix
    for a0,s0,a1,s1 in itertools.product(range(m),range(n),range(m),range(n)):
        t[a0*n+s0,a1*n+s1] = transition[a0,s0,s1] * policy[s1,a1]
    return t
