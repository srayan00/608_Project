import numpy as np 
import torch
from ToyEnv import ToyEnv
from PosteriorSamplerGMM import GibbsSamplerGMM, HMCpymcGMM, VIpymcGMM
import collections
import time



class PSRL:
    def __init__(self,  env = ToyEnv(), sampler = GibbsSamplerGMM, n_samples = 1000, n_iterations=50000, H = 100, T = 10): # Add a sampler argument
        self.n_samples = n_samples
        self.H = H
        self.T = T
        self.history = np.zeros((T*H, 4)) # (s, a, r, s')
        self.env = env
        self.currpolicy = None
        self.sampler = sampler
        self.history_regret = []
        self.history_rewards = []
        self.episode_rewards = []
        self.episode_regret = []
        self.start_time = time.time()
        self.end_time = None
        
        
        # Getting optimal policies for the true model
        self.Vstr, self.Qstr, self.policy_str = self.env.solve_optimal_policy(self.H)
        
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
                sampler = self.sampler(n_samples = self.n_samples, n_components = self.env.true_k) #, alpha = self.alpha0[s, a], mu0 = self.mu0[s, a], sigma0 = self.sigma0[s, a], alphaG = self.alphaG[s, a], betaG = self.betaG[s, a])
                sampler.fit(relevant_hist)
                reward_post_params[s, a] = sampler.sample_posterior()
        return reward_post_params
    
    def run(self):
        # episode_rewards = []
        # episode_regret = []
        self.start_time = time.time()
        
        for t in range(self.T):
            print(f"Iteration: {t}")
            transitions = self.sample_transition_dynamics() # first step
            reward_samples = self.sample_reward_posteriors(t)
            expected_rewards = np.zeros((self.env.nS, self.env.nA))
            for s in range(self.env.nS):
                for a in range(self.env.nA):
                    expected_rewards[s, a] = reward_samples[s, a, 0].dot(reward_samples[s, a, 2])
            # V, Q, policy = self.backward_induction(expected_rewards, transitions, self.H)
            V, Q, policy = self.backward_induction(expected_rewards, self.env.P, self.H)

            regret = self.env.initial_state_dist.dot(self.Vstr[:, 0] - V[:, 0])
            
            print(f"Episode {t} Regret: {regret}")
            self.episode_regret.append(regret)
            self.policy_evaluation(t)
            self.update_transition_dynamics(t)
            
        self.V = V
        self.Q = Q
        self.currpolicy = policy
        self.end_time = time.time()
        
        return V, Q, policy
    
    def to_dict(self):
        d = {}
        d["H"] = self.H
        d["T"] = self.T
        d["policy"] = self.currpolicy
        d["V"] = self.V
        d["Q"] = self.Q
        d["episode_regret"] = self.episode_regret
        d["time_taken"] = self.end_time - self.start_time
        return d
                    
    
if __name__ == "__main__":
    # Initialize parameters
    # new_env = ToyEnv(3, 3, 2)
    # init_state = new_env.reset()
    # alg = PSRL(env=new_env, sampler=HMCpymcGMM, T=1)
    # _, _, policy_g = alg.run()
    
    # print(alg.episode_regret)
    
    
    new_env = ToyEnv(3, 3, 2)
    init_state = new_env.reset()
    alg = PSRL(env=new_env, sampler=VIpymcGMM, T=1)
    _, _, policy_g = alg.run()
    
    print(alg.episode_regret)
        
        
    # env = ToyEnv()
    # psrl_g = PSRL()
    # init_state = env.reset()
    # transitions = env.P
    # expected_rewards = np.zeros((env.nS, env.nA))
    # for s in range(env.nS):
    #     for a in range(env.nA):
    #         expected_rewards[s, a] = env._compute_means(s, a).dot(env.true_pi)
    
    # # Compute the optimal value function and policy
    # Vg, Qg, policy_g = psrl_g.backward_induction(expected_rewards, transitions, 10)
    
    # print(policy_g)
    
    # psrl_h = PSRL(sampler=HMCpymcGMM)
    # init_state = env.reset()
    # transitions = env.P
    # expected_rewards = np.zeros((env.nS, env.nA))
    # for s in range(env.nS):
    #     for a in range(env.nA):
    #         expected_rewards[s, a] = env._compute_means(s, a).dot(env.true_pi)
            
    # Vh, Qh, policy_h = psrl_h.backward_induction(expected_rewards, transitions, 10)
    # print(policy_h)
