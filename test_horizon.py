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
    list_of_horizons_time = [(10, 100), (25, 40), (50, 20), (100, 10), (200, 5)]
    sample_dict = sample_dict = {"Gibbs": {"horizons": {10: 0, 25:0, 50: 0, 100:0, 200:0}}, "HMC": {"horizons": {10: 0, 25:0, 50: 0, 100:0, 200:0}}}
    for h, t in list_of_horizons_time:
        print("Running for H: ", h)
        new_env = ToyEnv.ToyEnv(4, 3, 2, seed)
        with open(f"results/horizons/env_{seed}_horizon{h}.pkl", "wb") as f:
            pickle.dump(new_env.to_dict(), f)
        init_state = new_env.reset()

        print(f"Running Gibbs Sampler for {h}")
        st_time_g = time.time()
        alg_g = psrl.PSRL(env=ToyEnv.ToyEnv(4, 3, 2, seed), sampler=GibbsSamplerGMM, H =h, T = t)
        _, _, policy_g = alg_g.run()
        end_time_g = time.time()
        with open(f"results/horizons/{seed}_horizon{h}_G.pkl", "wb") as g:
            pickle.dump(alg_g.to_dict(), g)
        sample_dict["Gibbs"]["horizons"][h] = end_time_g - st_time_g

        init_state = new_env.reset()
        st_time_h = time.time()
        print(f"Running HMC Sampler for {h}")
        alg_h = psrl.PSRL(env=ToyEnv.ToyEnv(4, 3, 2, seed), sampler=HMCpymcGMM, H = h)
        _, _, policy_h = alg_h.run()
        end_time_h = time.time()
        with open(f"results/horizons/{seed}_horizon{h}_H.pkl", "wb") as h:
            pickle.dump(alg_h.to_dict(), h)
        sample_dict["HMC"]["horizons"][h] = end_time_h - st_time_h
    with open(f"time_comps_horizons_{seed}.json", "w") as f:
        json.dump(sample_dict, f)