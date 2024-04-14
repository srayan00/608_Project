import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import collections
import torch

class PosteriorSamplerGMM:
    def __init__(self, n_samples, n_components):
        self.n_samples = n_samples
        self.n_components = n_components
    
    def fit(self, X):
        pass

class GibbsSamplerGMM(PosteriorSamplerGMM):
    def __init__(self, n_samples, n_components, alpha = None, mu0 = 0, sigma0 = 1, alphaG = 1, betaG = 0.5):
        super().__init__(n_samples, n_components)
        # Initialize prior parameters
        if alpha is None:
            self.alpha = np.ones(n_components)/n_components
        else:
            self.alpha = alpha

        # Normal on mu
        self.mu0 = mu0
        self.sigma0 = sigma0

        # Inverse gamma on sigma
        self.alphaG = alphaG
        self.betaG = betaG # scale paramterization

        # Tracking parameters
        self.samples = []
        self.currmu = None
        self.currsigma = None
        self.currpi = None
        self.currZ = None

    def sample_prior(self):
        self.currmu = np.random.normal(self.mu0, self.sigma0, self.n_components)
        self.currsigma = 1/np.random.gamma(self.alphaG, 1/self.betaG, self.n_components)
        self.currpi = np.random.dirichlet(self.alpha)
    
    def sample_mixing_probs(self, Z):
        counter = collections.Counter(Z)
        counts = np.array([counter[k] for k in range(self.n_components)])
        self.currpi = np.random.dirichlet(self.alpha + counts)
    
    def sample_mu(self, Z, X):
        for k in range(self.n_components):
            X_k = X[Z == k]
            n_k = X_k.shape[0]
            sample_mean = np.mean(X_k) if n_k > 0 else self.mu0
            sample_var = 1/(n_k/self.currsigma[k] + 1/self.sigma0)
            sample_mean = sample_var * (self.mu0/self.sigma0 + n_k * sample_mean/self.currsigma[k])
            self.currmu[k] = np.random.normal(sample_mean, sample_var)
    
    def sample_sigma(self, Z, X):
        for k in range(self.n_components):
            X_k = X[Z == k]
            n_k = X_k.shape[0]
            shifted_X_k = X_k - self.currmu[k]
            new_alpha = np.dot(shifted_X_k.T, shifted_X_k)/2 + self.alphaG
            new_scale = self.betaG + n_k/2
            self.currsigma[k] = 1/np.random.gamma(new_alpha, 1/new_scale)
    
    def sample_assignment(self, xi):
        probs = np.zeros(self.n_components)
        for k in range(self.n_components):
            probs[k] = self.currpi[k] * stats.norm.pdf(xi, loc=self.currmu[k], scale=self.currsigma[k])
        probs = probs/np.sum(probs)
        return np.random.choice(self.n_components, p=probs)
    
    def sample_assignments(self, X):
        Z = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            Z[i] = self.sample_assignment(X[i])
        self.currZ = Z

    def update_history(self):
        self.samples.append((self.currmu, self.currsigma, self.currpi))

    def fit(self, X):
        if X is None:
            for _ in range(self.n_samples):
                self.sample_prior()
                self.update_history()
        else:
            self.sample_prior()
            self.sample_assignments(X)
            for _ in range(self.n_samples):
                self.sample_mixing_probs(self.currZ)
                self.sample_mu(self.currZ, X)
                self.sample_sigma(self.currZ, X)
                self.sample_assignments(X)
                self.update_history()
        return self.samples
    
class HamiltonianSamplerGMM:
    def __init__(self, n_samples, n_leapfrog_steps, t_delta, eta):
        self.n_samples = n_samples
        self.n_leapfrog_steps = n_leapfrog_steps
        self.t_delta = t_delta
        self.eta = eta

        self.n_components = n_components

        # Initialize prior parameters
        self.alpha = torch.ones(self.n_components)/self.n_components

        # Normal on mu
        self.mu0 = 0
        self.sigma0 = 1

        # Inverse gamma on sigma
        self.alphaG = 1
        self.betaG = 0.5 # scale paramterization

        # Tracking parameters
        self.samples = []

        # mu, sigma, pi concatenated
        self.currmu = None
        self.currsigma = None
        self.currpi = None
        self.currloglik = None
    
    def sample_prior(self):
        self.currmu = torch.normal(self.mu0, self.sigma0, self.n_components, requires_grad=True)
        self.currsigma = 1/torch.gamma(self.alphaG, 1/self.betaG, self.n_components, requires_grad=True)
        self.currpi = torch.dirichlet(self.alpha, requires_grad=True)

    def normal_pdf(x, mu, sigma):
        return 1/(torch.sqrt(2 * np.pi* sigma)) * torch.exp(-0.5 * ((x - mu))**2/sigma)
    
    def log_prob(X):
        loglik = 0
        for i in range(len(X)):
            xi = X[i]
            likelihood = 0
            for k in range(self.n_components):
                likelihood += self.currpi[k] * normal_pdf(xi, self.currmu[k], self.currsigma[k])
            loglik += torch.log(likelihood)
        return loglik
    
    def hamiltonian(X):
        p_mu = torch.normal(0, 1, self.n_components)
        p_sigma = torch.gamma(self.alphaG, 1/self.betaG, self.n_components)
        p_pi = torch.dirichlet(self.alpha)
        H = -log_likelihood(X) + 0.5 * torch.sum(p_mu**2) + 0.5 * torch.sum(p_sigma**2) + 0.5 * torch.sum(p_pi**2)
        return H


    

    
