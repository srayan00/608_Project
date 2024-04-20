import numpy as np
import ToyEnv
import psrl
from PosteriorSamplerGMM import GibbsSamplerGMM, HMCpymcGMM
import time
import json
import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the PSRL algorithm on the ToyEnv environment')
parser.add_argument('--seed', type=int, default=43, help='The seed for the random number generator')

# Seed 1 = 42, Seed 2 = 43, Seed 3 = 44

if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    list_of_nS = [2, 4, 6, 8, 10]
    sample_dict = sample_dict = {"Gibbs": {"state": {2: 0, 4: 0, 6:0, 8:0, 10:0}}, "HMC": {"state": {2: 0, 4: 0, 6:0, 8:0, 10:0}}}
    for nS in list_of_nS:
        print("Running for nS: ", nS)
        new_env = ToyEnv.ToyEnv(nS, 3, 2, seed)
        with open(f"results/states/env_{seed}_state{nS}.pkl", "wb") as f:
            pickle.dump(new_env.to_dict(), f)
        init_state = new_env.reset()
        st_time_g = time.time()

        print(f"Running Gibbs Sampler for {nS}")
        alg_g = psrl.PSRL(env=ToyEnv.ToyEnv(nS, 3, 2, seed), sampler=GibbsSamplerGMM)
        _, _, policy_g = alg_g.run()
        end_time_g = time.time()
        with open(f"results/states/{seed}_state{nS}_G.pkl", "wb") as g:
            pickle.dump(alg_g.to_dict(), g)
        # np.save(f"results/states/{seed}_{nS}_policy_g.npy", policy_g)
        sample_dict["Gibbs"]["state"][nS] = end_time_g - st_time_g

        init_state = new_env.reset()
        st_time_h = time.time()
        print(f"Running HMC Sampler for {nS}")
        alg_h = psrl.PSRL(env=ToyEnv.ToyEnv(nS, 3, 2, seed), sampler=HMCpymcGMM)
        _, _, policy_h = alg_h.run()
        end_time_h = time.time()
        with open(f"results/states/{seed}_state{nS}_H.pkl", "wb") as h:
            pickle.dump(alg_h.to_dict(), h)
        sample_dict["HMC"]["state"][nS] = end_time_h - st_time_h
    with open(f"time_{seed}.json", "w") as f:
        json.dump(sample_dict, f)