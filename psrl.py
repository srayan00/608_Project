import numpy as np 
import torch
from ToyEnv import ToyEnv
import PosteriorSamplerGMM
import collections


class PSRL:
    def __init__(self,  env = ToyEnv(), n_samples = 1000, H = 100, T = 10): # Add a sampler argument
        self.n_samples = n_samples
        self.H = H
        self.T = T
        self.history = np.zeros((T*H, 4)) # (s, a, r, s')
        self.env = env
        self.currpolicy = None

        # Priors for transition matrix
        self.P0 = np.ones((env.nS, env.nA, env.nS))/env.nS

        # Priors for reward distribution
        # self.mu0 = np.zeros((env.nS, env.nA, env.true_k))
        # self.sigma0 = np.ones((env.nS, env.nA, env.true_k))
        # self.alpha0 = np.ones((env.nS, env.nA, env.true_k))/env.true_k
        # self.alphaG = np.ones((env.nS, env.nA))
        # self.betaG = np.ones((env.nS, env.nA))/2


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
        policy_to_call = self.currpolicy[curr_state]
        history = np.zeros((self.H, 4))
        for h in range(self.H):
            action = policy_to_call[h]
            next_state, reward = self.env.step(action)
            history[h] = np.array([curr_state, action, reward, next_state])
            curr_state = next_state
        self.history[t*self.H:(t+1)*self.H] = history

    def sample_transition_dynamics(self):
        transitions = {}
        for s in range(self.env.nS):
            transitions[s] = {a : [] for a in range(self.env.nA)}
            for a in self.env.action_space:
                transitions[s][a] = np.random.dirichlet(self.P0[s, a])
        return transitions

    def update_transition_dynamics(self, t):
        history = self.history[:t*self.H]
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                relevant_hist = history[(history[:, 0] == s) & (history[:, 1] == a)]
                counter = collections.Counter(relevant_hist[:, 3])
                counts = np.array([counter[k] for k in range(self.env.nS)])
                self.P0[s, a] = self.P0[s, a] + counts
    
    def sample_reward_posteriors(self, t):
        reward_post_params = np.zeros((self.env.nS, self.env.nA, 3, self.env.true_k))
        history = self.history[:t*self.H]
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                if history.size == 0:
                    relevant_hist = None
                else:
                    relevant_hist = history[(history[:, 0] == s) & (history[:, 1] == a), 2]
                sampler = PosteriorSamplerGMM.GibbsSamplerGMM(n_samples = self.n_samples, n_components = self.env.true_k) #, alpha = self.alpha0[s, a], mu0 = self.mu0[s, a], sigma0 = self.sigma0[s, a], alphaG = self.alphaG[s, a], betaG = self.betaG[s, a])
                samples = sampler.fit(relevant_hist)
                reward_post_params[s, a] = samples[np.random.choice(self.n_samples, size = 1)]
        return reward_post_params
    
    def run(self):
        for t in range(self.T):
            print(f"Iteration: {t}")
            transitions = self.sample_transition_dynamics() # first step
            reward_samples = self.sample_reward_posteriors(t)
            expected_rewards = np.zeros((self.env.nS, self.env.nA))
            for s in range(self.env.nS):
                for a in range(self.env.nA):
                    expected_rewards[s, a] = reward_samples[s, a, 0].dot(reward_samples[s, a, 2])
            V, Q, policy = self.backward_induction(expected_rewards, transitions, self.H)
            self.policy_evaluation(t)
            self.update_transition_dynamics(t)
        return V, Q, policy
                
    


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