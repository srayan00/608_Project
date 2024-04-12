import numpy as np 
import torch
from ToyEnv import ToyEnv
import PosteriorSamplerGMM


class PSRL:
    def __init__(self, sampler = None, H = 10, T = 10, env = ToyEnv()):
        self.post_sampler = sampler
        self.H = H
        self.T = T
        self.history = np.zeros((T*H, 4)) # (s, a, r, s')
        self.env = env
        self.currpolicy = None

        # Priors for transition matrix
        self.P0 = np.ones((env.nS, env.nA, env.nS))/env.nS

        # Priors for reward distribution
        self.mu0 = np.zeros((env.nS, env.nA, env.true_k))
        self.sigma0 = np.ones((env.nS, env.nA, env.true_k))
        self.alpha0 = np.ones((env.nS, env.nA, env.true_k))/env.true_k
        self.alphaG = np.ones((env.nS, env.nA))
        self.betaG = np.ones((env.nS, env.nA))/2


    def backward_induction(self, expected_means, transitions, H):
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
        self.currpolicy = policy
        return V, Q, policy
    
    def policy_evaluation(self, t):
        curr_state = self.env.reset()
        policy_to_call = self.currpolicy[init_state]
        history = np.zeros((self.H, 4))
        for h in range(self.H):
            action = policy_to_call[h]
            next_state, reward = self.env.step(action)
            history[h] = np.array([curr_state, action, reward, next_state])
            curr_state = next_state
        self.history[t*self.H:(t+1)*self.H] = history

    

    def update_transition_dynamics(self):
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                relevant_hist = self.history[self.history[:, 0] == s and self.history[:, 1] == a]
                self.P0[s, a] = self.P0[s, a] + np.histogram(relevant_hist[:, 3], bins = np.arange(self.env.nS))[0]
    
    def update_reward_posteriors(self):
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                relevant_hist = self.history[self.history[:, 0] == s and self.history[:, 1] == a, 2]
                sampler = PosteriorSamplerGMM.GibbsSamplerGMM(n_samples = 100, n_components = self.env.true_k, alpha = self.alpha0[s, a], mu0 = self.mu0[s, a], sigma0 = self.sigma0[s, a], alphaG = self.alphaG[s, a], betaG = self.betaG[s, a])
                samples_hist = sampler.fit(relevant_hist)
                
    


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