import numpy as np 
import torch
from ToyEnv import ToyEnv


class PSRL:
    def __init__(sampler = None, h = 10, T = 10, env = ToyEnv()):
        self.post_sampler = sampler
        self.h = h
        self.T = T
        self.history = np.zeros((T*h, 4)) # (s, a, r, s')
        self.env = env
        self.policy = None

        # Priors for transition matrix
        self.P0 = np.ones((env.nS, env.nA, env.nS))/env.nS

        # Priors for reward distribution
        self.mu0 = np.zeros((env.nS, env.nA, env.true_k))
        self.sigma0 = np.ones((env.nS, env.nA, env.true_k))
        self.alpha0 = np.ones((env.nS, env.nA, env.true_k))/env.true_k
        self.alphaG = np.ones((env.nS, env.nA))
        self.betaG = np.ones((env.nS, env.nA))/2


    def backward_induction(expected_means, transitions, H):
        """
        Compute the optimal value function and policy using backward induction.
        """
        num_states, num_actions = expected_means.shape
        policy = np.zeros((num_states, H), dtype=int)
        Q = np.zeros((num_states, num_actions, H))
        Q[:, :, H - 1] = expected_means
        V = np.zeros((num_states, H))
        V[:, H - 1] = np.max(Q[:, :, H - 1], axis=1)
        for h in range(H - 2, -1, -2):
            for s in range(num_states):
                for a in range(num_actions):
                    Q[s, a, h] = expected_means[s, a] + transitions[s][a].dot(V[:, h + 1])
            V[:, h] = np.max(Q[:, :, h], axis=1)
            policy[:, h] = np.argmax(Q[:, :, h], axis=1)
        return V, Q, policy

    def update_transition_dynamics():
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                relevant_hist = self.history[self.history[:, 0] == s and self.history[:, 1] == a]
                self.P0[s, a] = self.P0[s, a] + np.histogram(relevant_hist[:, 3], bins = np.arange(self.env.nS))[0]
    
    def update_reward_posteriors():
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                relevant_hist = self.history[self.history[:, 0] == s and self.history[:, 1] == a]
                
    


if __name__ == "__main__":
    # Initialize parameters
    env = ToyEnv()
    psrl = PSRL()
    init_state = env.reset()
    transitions = env.P
    expected_rewards = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            expected_rewards[s, a] = env._compute_means(s, a).dot(env.true_pi)
    
    # Compute the optimal value function and policy
    V, Q, policy = psrl.backward_induction(expected_rewards, transitions, 10)

    print(policy[1])