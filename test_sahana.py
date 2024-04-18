import numpy as np
import ToyEnv
import psrl
from PosteriorSamplerGMM import GibbsSamplerGMM, HMCpymcGMM
import time

if __name__ == "__main__":
    list_of_nS = [2, 3, 4]
    sample_dict = sample_dict = {"Gibbs": {"state": {2: 0, 3: 0, 4: 0}}, "HMC": {"state": {2: 0, 3: 0, 4: 0}}}
    for nS in list_of_nS:
        new_env = ToyEnv.ToyEnv(nS, 3, 2)
        init_state = new_env.reset()
        st_time_g = time.time()
        alg_g = psrl.PSRL(env=new_env, sampler=GibbsSamplerGMM)
        _, _, policy_g = alg_g.run()
        end_time_g = time.time()
        np.save(f"results/states/{nS}_policy_g.npy", policy_g)
        sample_dict["Gibbs"]["state"][nS] = end_time_g - st_time_g

        init_state = new_env.reset()
        st_time_h = time.time()
        alg_h = psrl.PSRL(env=new_env, sampler=HMCpymcGMM)
        _, _, policy_h = alg_h.run()
        end_time_h = time.time()
        np.save(f"results/states/{nS}_policy_h.npy", policy_h)
        sample_dict["HMC"]["state"][nS] = end_time_h - st_time_h

