import numpy as np
import ToyEnv
import psrl
import PosteriorSamplerGMM

if __name__ == "__main__":
    new_env = ToyEnv.ToyEnv(3, 3, 2)
    init_state = new_env.reset()
    alg = psrl.PSRL(new_env)
    _, _, policy = alg.run()