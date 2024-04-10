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
    
    def sample_mu(Z, X, k):
        X_k = X[Z == k]
        n_k = X_k.shape[0]
        sample_mean = np.mean(X_k) if n_k > 0 else self.mu0
        sample_var = 1/(n_k/self.currsigma[k] + 1/self.sigma0)
        sample_mean = sample_var * (self.mu0/self.sigma0 + n_k * sample_mean/self.currsigma[k])
        return np.random.normal(sample_mean, sample_var)
    
    def sample_sigma(Z, X, k):
        X_k = X[Z == k]
        n_k = X_k.shape[0]
        shifted_X_k = X_k - self.currmu[k]
        new_alpha = np.dot(shifted_X_k.T, shifted_X_k)/2 + self.alphaG
        new_scale = self.betaG + n_k/2
        return 1/np.random.gamma(new_alpha, 1/new_scale)
    
