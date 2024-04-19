import numpy as np
import ToyEnv
import psrl
from PosteriorSamplerGMM import GibbsSamplerGMM, HMCpymcGMM
import time
import json
import argparse

parser = argparse.ArgumentParser(description='Run the PSRL algorithm on the ToyEnv environment')
parser.add_argument('--seed', type=int, default=43, help='The seed for the random number generator')

# Seed 1 = 42, Seed 2 = 43, Seed 3 = 44

if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    list_of_ncomps = [2, 3, 4, 5]
    sample_dict = sample_dict = {"Gibbs": {"ncomps": {2: 0, 3: 0, 4:0, 5:0}}, "HMC": {"ncomps": {2: 0, 3: 0, 4:0, 5:0}}}
    for k in list_of_ncomps:
        new_env = ToyEnv.ToyEnv(4, 3, k)
        init_state = new_env.reset()
        st_time_g = time.time()
        alg_g = psrl.PSRL(env=new_env, sampler=GibbsSamplerGMM)
        _, _, policy_g = alg_g.run()
        end_time_g = time.time()
        np.save(f"results/ncomps/{seed}_{k}_policy_g.npy", policy_g)
        sample_dict["Gibbs"]["ncomps"][k] = end_time_g - st_time_g

        init_state = new_env.reset()
        st_time_h = time.time()
        alg_h = psrl.PSRL(env=new_env, sampler=HMCpymcGMM)
        _, _, policy_h = alg_h.run()
        end_time_h = time.time()
        np.save(f"results/ncomps/{seed}_{k}_policy_h.npy", policy_h)
        sample_dict["HMC"]["ncomps"][k] = end_time_h - st_time_h
    with open(f"time_comps_{seed}.json", "w") as f:
        json.dump(sample_dict, f)