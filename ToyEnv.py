import numpy as np
import itertools

class ToyEnv:
    def __init__(self, nS = 4, nA = 3, truek = 3, seed = None):
        # Define Action space and Observation space
        self.nS = nS
        self.nA = nA
        self.seed = seed
        self.rng = np.random.default_rng(seed = seed)

        self.action_space = np.arange(nA)
        self.observation_space = np.arange(nS)

        # Define initial state distribution
        self.initial_state_dist = self.__initialstate_dist()

        # Define Transition Probability
        self.P = {}
        for s in range(self.nS):
            self.P[s] = {a : [] for a in self.action_space}
            for a in self.action_space:
                self.P[s][a] = self._calculate_transition_prob(s, a)

        # Define Reward distribution where the state and action dictate the mean of the distributions
        self.true_k = truek
        self.true_pi = self.__compute_pi()
        self.true_Sigma = np.arange(self.true_k, 0, -1)
        self.true_Sigma = np.ones(self.true_k)
    
    def __initialstate_dist(self):
        dist = np.ones(self.nS)
        dist[0] += 3
        dist[1] += 1
        dist = dist/np.sum(dist)
        return dist
    
    def __compute_pi(self):
        dist = np.ones(self.true_k)
        dist[-1] += 3
        return dist/np.sum(dist)


    def _compute_means(self, state, action):
         mus = np.arange(0, 3*self.true_k, 3) + state + action
         return mus
     
    def _compute_expected_reward(self):
        nS, nA = self.nS, self.nA
        S, A = self.observation_space, self.action_space
        
        expected_rewards = (np.array([self._compute_means(s,a) 
                                      for s,a in 
                                      itertools.product(S, A)]) @ self.true_pi
                            ).reshape(nS,-1)
        
        return expected_rewards


    def _get_reward(self, state, action):
        mus = self._compute_means(state, action)
        z = self.rng.choice(self.true_k, p=self.true_pi)
        x = self.rng.normal(mus[z], self.true_Sigma[z])
        return x, z
    
    def _calculate_transition_prob(self, curr_state, action_taken):
        probs = np.ones(self.nS)
        probs[curr_state] += 1
        if curr_state + action_taken >= self.nS:
             index = self.nS - 1
             next_index = self.nS - 1
        elif curr_state + action_taken + 1 >= self.nS:
            index = curr_state + action_taken
            next_index = self.nS - 1
        else:
            index = curr_state + action_taken
            next_index = curr_state + action_taken + 1
        probs[index] += 3
        probs[next_index] += 2
        probs = probs/np.sum(probs)
        return probs
    
    def reset(self):
        self.state = self.rng.choice(self.observation_space, p=self.initial_state_dist)
        return self.state
    
    def step(self, action):
        transitions = self.P[self.state][action]
        next_state = self.rng.choice(self.observation_space, p=[t for t in transitions])
        reward, _ = self._get_reward(next_state, action)
        self.s = next_state
        return next_state, reward
    
    def solve_optimal_policy(self, H):
        """Returns optimal policy and value functions for given environment using backwards induction 

        Args:
            H (int): Horizon for simulation
            
        Returns:True optimal value function, q function, and policy
        """
        
        num_states, num_actions = self.nS, self.nA 
        policy = np.zeros((num_states, H), dtype=int)
        
        expected_rewards = self._compute_expected_reward()
        
        Q = np.zeros((num_states, num_actions, H))
        Q[:, :, H - 1] = expected_rewards
        V = np.zeros((num_states, H))
        V[:, H - 1] = np.max(Q[:, :, H - 1], axis=1)
        for h in range(H - 2, -1, -2):
            for s in range(num_states):
                for a in range(num_actions):
                    Q[s, a, h] = expected_rewards[s, a] + self.P[s][a].dot(V[:, h + 1])
            V[:, h] = np.max(Q[:, :, h], axis=1)
            policy[:, h] = np.argmax(Q[:, :, h], axis=1)
        self.currpolicy = policy
        return V, Q, policy
    
    def to_dict(self):
        d = {}
        d['nS'] = self.nS
        d['nA'] = self.nA
        d['seed'] = self.seed
        d['truek'] = self.true_k
        d['initial_state_dist'] = self.initial_state_dist.tolist()
        d['true_pi'] = self.true_pi.tolist()
        d['true_Sigma'] = self.true_Sigma.tolist()
        return d


