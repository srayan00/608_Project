import numpy as np 
import torch
from ToyEnv import ToyEnv


class PSRL:
    def __init__(sampler):
        self.post_sampler = sampler


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

    # def update_transition_dynamics(transitions, policy, history):


if __name__ == "__main__":
    # Initialize parameters
    env = ToyEnv()
    init_state = env.reset()
    print(init_state)
    transitions = env.P
    expected_rewards = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            expected_rewards[s, a] = env._compute_means(s, a).dot(env.true_pi)
    
    # Compute the optimal value function and policy
    V, Q, policy = backward_induction(expected_rewards, transitions, 10)

    print(policy[1])