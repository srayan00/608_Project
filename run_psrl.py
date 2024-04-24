import numpy as np
import ToyEnv
import psrl
from PosteriorSamplerGMM import GibbsSamplerGMM, HMCpymcGMM
from VariationalInference import VIpymcGMM
import time
import json
import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the PSRL algorithm on the ToyEnv environment')
parser.add_argument('--seed', type=int, default=43, help='The seed for the random number generator')
parser.add_argument('--sampler', type=str, default="Gibbs", 
                    choices=['Gibbs', 'HMC', 'VI'],
                    help='The sampler used to sample from the posterior')
parser.add_argument('--nS', type=int, default=None, help='The number of states in the environment')
parser.add_argument('--nA', type=int, default=None, help='The number of actions in the environemnt')
parser.add_argument('--nC', type=int, default=None, help='The number of components in the environemnt')
parser.add_argument('--nE', type=int, default=None, help='The number of episodes in the environemnt')
parser.add_argument('--nH', type=int, default=None, help='The horizon in the environemnt')

samplers = {
    "VI" : VIpymcGMM,
    "Gibbs": GibbsSamplerGMM,
    "HMC" : HMCpymcGMM
}


if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    sampler = args.sampler
    nS = args.nS
    nA = args.nA
    nC = args.nC
    nE = args.nE
    nH = args.nH
    
    if nS is None:
        state = ""
        nS = 3
    else:
        state = f"_state{nS}"
        print("Running for state: ", nS)
        
    if nA is None:
        action = ""
        nA = 2
    else:
        action = f"_action{nA}"
        print("Running for action: ", nA)
        
    if nC is None:
        components = ""
        nC = 3
    else:
        components = f"_components{nC}"
        print("Running for components: ", nC)
        
    if nE is None:
        episodes = ""
        nE = 10
    else:
        episodes = f"_episodes{nE}"
        print("Running for episodes: ", nE)
        
    if nH is None:
        horizons = ""
        nH = 10
    else:
        horizons = f"_horizons{nH}"
        print("Running for horizons: ", nH)
    
    if nS is None and nA is None and nC is None and nE is None and nH is None and nH is None:
        print(f"Running default values: nS = {nS}, nA = {nA}, nC = {nC}, nE = {nE}, nH = {nH}")
    
    save_path = f"results/states/env_{seed}_{state}{action}{components}{episodes}{horizons}.pkl"
    
    
    new_env = ToyEnv.ToyEnv(nS, nA, nC, seed)
    with open(f"results/states/env_{seed}_state{nS}.pkl", "wb") as f:
        pickle.dump(new_env.to_dict(), f)
    init_state = new_env.reset()
    st_time_g = time.time()
    
    print(f"Running {sampler}")
    alg_g = psrl.PSRL(env=ToyEnv.ToyEnv(nS, nA, nC, seed), sampler=samplers[sampler])
    _, _, policy_g = alg_g.run()
    end_time_g = time.time()
    
    results_path = f"results/states/{seed}_{state}{action}{components}{episodes}{horizons}_{sampler}.pkl"
    with open(results_path, "wb") as g:
        pickle.dump(alg_g.to_dict(), g)
    
    