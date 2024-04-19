import numpy as np
from ToyEnv import ToyEnv
import os 
import seaborn as sns 
import matplotlib.pyplot as plt
import argparse
import re 


parser = argparse.ArgumentParser(description='Run regret evaluation for the HMC and Gibbs samplers')

parser.add_argument('--policy_path', type=str, default=None, help='Path where policies are stored.')
parser.add_argument('--save_path', type=str, default=None, help='Path where policies are stored.')


if __name__ == "__main__":
    args = parser.parse_args()
    policy_path = args.policy_path
    save_path = args.save_path
    # Load the data
    if policy_path is None:
        files = os.listdir("results/states")
    else:
        files = os.listdir(policy_path)
    
    if save_path is None:
        save_path = "results/regret"
        
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    file = "results/states/42_8_policy_h.npy"
    policy_h = np.load("results/states/42_8_policy_h.npy")
    policy_g = np.load("results/states/42_8_policy_g.npy")
    
    regret_h = []
    regret_g = []
    
    seed, nS = [int(x) for x in re.findall(r'(\d+)', file)]
    
    H = policy_g.shape[1]
    
    env = ToyEnv(nS, 3, 2)
    
    _, _, opt_policy = env.solve_optimal_policy(H)
    state = env.reset()
    
    optimal_reward = []
    for t in range(H):
        action = opt_policy[state,t]
        state, reward = env.step(action)
        optimal_reward.append(reward)
      
    state = env.reset()  
    g_reward = []
    for t in range(H):
        action = policy_g[state,t]
        state, reward = env.step(action)
        g_reward.append(reward)
        
    state = env.reset()  
    h_reward = []
    for t in range(H):
        action = policy_h[state,t]
        state, reward = env.step(action)
        h_reward.append(reward)
        
        
    regret_g = np.cumsum(np.array(optimal_reward) - np.array(g_reward))
    regret_h = np.cumsum(np.array(optimal_reward) - np.array(h_reward)) 
    print(regret_h)
    
    plt.plot(range(H), regret_g, label="Gibbs")
    plt.plot(range(H), regret_h, label="HMC")
    plt.show()