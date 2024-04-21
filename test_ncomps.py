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
    list_of_ncomps = [6, 7, 8, 9]
    sample_dict = sample_dict = {"Gibbs": {"ncomps": {6: 0, 7:0, 8:0, 9:0}}, "HMC": {"ncomps": {6:0, 7:0, 8:0, 9:0}}}
    for k in list_of_ncomps:
        print("Running for k: ", k)
        new_env = ToyEnv.ToyEnv(4, 3, k, seed)
        with open(f"results/ncomps/env_{seed}_ncomp{k}.pkl", "wb") as f:
            pickle.dump(new_env.to_dict(), f)
        init_state = new_env.reset()

        print(f"Running Gibbs Sampler for {k}")
        st_time_g = time.time()
        alg_g = psrl.PSRL(env=ToyEnv.ToyEnv(4, 3, k, seed), sampler=GibbsSamplerGMM)
        _, _, policy_g = alg_g.run()
        end_time_g = time.time()
        with open(f"results/ncomps/{seed}_ncomp{k}_G.pkl", "wb") as g:
            pickle.dump(alg_g.to_dict(), g)
        sample_dict["Gibbs"]["ncomps"][k] = end_time_g - st_time_g

        init_state = new_env.reset()
        st_time_h = time.time()
        print(f"Running HMC Sampler for {k}")
        alg_h = psrl.PSRL(env=ToyEnv.ToyEnv(4, 3, k, seed), sampler=HMCpymcGMM)
        _, _, policy_h = alg_h.run()
        end_time_h = time.time()
        with open(f"results/ncomps/{seed}_ncomp{k}_H.pkl", "wb") as h:
            pickle.dump(alg_h.to_dict(), h)
        sample_dict["HMC"]["ncomps"][k] = end_time_h - st_time_h
    with open(f"time_comps67_{seed}.json", "w") as f:
        json.dump(sample_dict, f)