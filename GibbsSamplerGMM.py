import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import collections

class GibbsSamplerGMM:
    def __init__(n_samples, n_components):
        self.n_samples = n_samples
        self.n_components = n_components
        self.burn_in = 100

        # Initialize prior parameters
        self.alpha = np.ones(n_components)/n_components

        # Normal on mu
        self.mu0 = 0
        self.sigma0 = 1

        # Inverse gamma on sigma
        self.alphaG = 1
        self.betaG = 0.5 # scale paramterization

        # Tracking parameters
        self.samples = []
        self.currmu = None
        self.currsigma = None
        self.currpi = None
        self.currZ = None

    def sample_prior():
        mu = np.random.normal(self.mu0, self.sigma0, self.n_components)
        sigma = np.random.gamma(self.alphaG, 1/self.betaG, self.n_components)
        pi = np.random.dirichlet(self.alpha)
        return mu, 1/sigma, pi
    
    def sample_mixing_probs(Z):
        counter = collections.Counter(Z)
        counts = np.array([counter[k] for k in range(self.n_components)])
        return np.random.dirichlet(self.alpha[:K] + counts)
    
    def sample_