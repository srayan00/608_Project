import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import collections

class GibbsSamplerGMM:
    def __init__(n_samples, n_components):
        self.n_samples = n_samples
        self.n_components = n_components

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
        self.currmu = np.random.normal(self.mu0, self.sigma0, self.n_components)
        self.currsigma = 1/np.random.gamma(self.alphaG, 1/self.betaG, self.n_components)
        self.currpi = np.random.dirichlet(self.alpha)
    
    def sample_mixing_probs(Z):
        counter = collections.Counter(Z)
        counts = np.array([counter[k] for k in range(self.n_components)])
        self.currpi = np.random.dirichlet(self.alpha + counts)
    
    def sample_mu(Z, X):
        for k in range(self.n_components):
            X_k = X[Z == k]
            n_k = X_k.shape[0]
            sample_mean = np.mean(X_k) if n_k > 0 else self.mu0
            sample_var = 1/(n_k/self.currsigma[k] + 1/self.sigma0)
            sample_mean = sample_var * (self.mu0/self.sigma0 + n_k * sample_mean/self.currsigma[k])
            self.currmu[k] = np.random.normal(sample_mean, sample_var)
    
    def sample_sigma(Z, X):
        for k in range(self.n_components):
            X_k = X[Z == k]
            n_k = X_k.shape[0]
            shifted_X_k = X_k - self.currmu[k]
            new_alpha = np.dot(shifted_X_k.T, shifted_X_k)/2 + self.alphaG
            new_scale = self.betaG + n_k/2
            self.currsigma[k] = 1/np.random.gamma(new_alpha, 1/new_scale)
    
    def sample_assigment(xi):
        probs = np.zeros(self.n_components)
        for k in range(self.n_components):
            probs[k] = self.currpi[k] * stats.norm.pdf(xi, loc=self.currmu[k], scale=self.currsigma[k])
        probs = probs/np.sum(probs)
        return np.random.choice(self.n_components, p=probs)
    
    def sample_assigments(X):
        Z = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            Z[i] = sample_assignment(X[i])
        self.currZ = Z

    def update_history():
        self.samples.append((self.currmu, self.currsigma, self.currpi, self.currZ))

    def fit(X):
        sample_prior()
        sample_assignments(X)
        for i in range(self.n_samples):
            sample_mixing_probs(self.currZ)
            sample_mu(self.currZ, X)
            sample_sigma(self.currZ, X)
            sample_assigments(X)
            update_history()
        return self.samples
    

    
