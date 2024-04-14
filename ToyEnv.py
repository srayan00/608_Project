import numpy as np

class ToyEnv:
    def __init__(self, nS = 4, nA = 3, truek = 3):
        # Define Action space and Observation space
        self.nS = nS
        self.nA = nA
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
        self.true_Sigma = np.ones(truek)/10
    
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
         mu0 = 2*state - action
         mu1 = state * action
         mu2 = state + action
         return np.array([mu0, mu1, mu2])


    def _get_reward(self, state, action):
        mus = self._compute_means(state, action)
        z = np.random.choice(self.true_k, p=self.true_pi)
        x = np.random.normal(mus[z], self.true_Sigma[z])
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
        self.state = np.random.choice(self.observation_space, p=self.initial_state_dist)
        return self.state
    
    def step(self, action):
        transitions = self.P[self.state][action]
        next_state = np.random.choice(self.observation_space, p=[t for t in transitions])
        reward, _ = self._get_reward(next_state, action)
        self.s = next_state
        return next_state, reward

         
        


