import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import collections
import torch
import pymc as pm 


class PosteriorSamplerGMM:
    def __init__(self, n_samples, n_components, burn_in=500,):
        self.n_samples = n_samples
        self.n_components = n_components
        self.burn_in = burn_in
        
    def fit(self, X):
        pass

class GibbsSamplerGMM(PosteriorSamplerGMM):
    def __init__(self, n_samples, n_components, burn_in=500,
                 alpha = None, mu0 = 0, sigma0 = 1, alphaG = 1, betaG = 2):
        super().__init__(n_samples, n_components, burn_in)
        
        
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
        self.betaG = betaG # rate paramterization

        # Tracking parameters
        self.samples = np.zeros((n_samples, 3, n_components))
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
            new_scale = np.dot(shifted_X_k.T, shifted_X_k)/2 + self.betaG
            new_alpha = self.alphaG + n_k/2
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

    def update_history(self, t):
        self.samples[t] = np.array([self.currmu, self.currsigma, self.currpi])

    def fit(self, X, verbose = False):
        if X is None or X.size == 0:
            if verbose:
                print("No data provided. Sampling from prior.")
            for t in range(self.n_samples):
                self.sample_prior()
                self.update_history(t)
        else:
            self.sample_prior()
            self.sample_assignments(X)
            for t in range(self.n_samples):
                self.sample_mixing_probs(self.currZ)
                self.sample_mu(self.currZ, X)
                self.sample_sigma(self.currZ, X)
                self.sample_assignments(X)
                self.update_history(t)
                 
    def sample_posterior(self):
        assert self.samples is not None
        sample_id = np.random.choice(range(self.burn_in, self.n_samples), size=1)[0]
        
        return self.samples[sample_id]
        
    
    
class HMCpymcGMM(PosteriorSamplerGMM):
    def __init__(self,n_samples, n_components, alpha=None, burn_in=500,
                 mu0=None, sigma0=None, alphaG=None, betaG=None,
                 n_leapfrog_steps=None, t_delta=None, eta=None):
        super().__init__(n_samples, n_components, burn_in)
        
        self.n_leapfrog_steps = n_leapfrog_steps
        self.t_delta = t_delta
        self.eta = eta
                
        # Initialize priors
        # dirichlet for mixing probs
        if alpha is None:
            self.alpha = np.ones(n_components)/n_components
        else:
            self.alpha = alpha
        
        # gaussian for mu
        if mu0 is None:
            self.mu0 = 0
        else: 
            self.mu0 = mu0
        
        if sigma0 is None:
            self.sigma0 = 1
        else:
            self.sigma0 = sigma0

        
        # Inverse gamma on sigma
        if alphaG is None:
            self.alphaG = 1
        else:
            self.alphaG = alphaG
            
        if betaG is None:
            self.betaG = 0.5 #
        else:
            self.betaG = betaG
            
        # Instantiate sampler 
        self.sampler = pm.Model(coords={"cluster" : range(n_components)}) 
        # Create pymc model 
        with self.sampler:
             self.mu = pm.Normal("mu",
                            mu = self.mu0, 
                            sigma = self.sigma0,
                            dims="cluster"
                            )
             self.sigma = pm.InverseGamma("sigma",
                                     beta = self.betaG,
                                     alpha = self.alphaG,
                                     dims = "cluster")
             
             self.weights = pm.Dirichlet("w", np.ones(n_components), dims="cluster")
             
             
    def fit(self, X):
        with self.sampler:
            obs = pm.NormalMixture("x",  w=self.weights,
                                   mu=self.mu, sigma=self.sigma, observed=X)
            idata = pm.sample(self.n_samples, step = pm.HamiltonianMC())
        
        self.post = idata.posterior
        
    def sample_posterior(self):
        # Taking sample from one of the chains 
        chain_id = np.random.choice(self.post.chain, 1)[0]
        sample_id = np.random.choice(range(self.burn_in, self.n_samples), size=1)[0]
        sample = np.array([self.post["mu"].values[chain_id, sample_id, :],
                           self.post["sigma"].values[chain_id, sample_id, :],
                           self.post["w"].values[chain_id, sample_id, :]])
        return sample
            
        
class VIpymcGMM(PosteriorSamplerGMM):
    def __init__(self,n_samples, n_components,iterations=50000,burn_in=500, alpha=None,
                 mu0=None, sigma0=None, alphaG=None, betaG=None, seed = None):
        super().__init__(n_samples, n_components, burn_in=burn_in)
        
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        # Initialize priors
        # dirichlet for mixing probs
        if alpha is None:
            self.alpha = np.ones(n_components)/n_components
        else:
            self.alpha = alpha
        
        # gaussian for mu
        if mu0 is None:
            self.mu0 = 0
        else: 
            self.mu0 = mu0
        
        if sigma0 is None:
            self.sigma0 = 1
        else:
            self.sigma0 = sigma0

        
        # Inverse gamma on sigma
        if alphaG is None:
            self.alphaG = 1
        else:
            self.alphaG = alphaG
            
        if betaG is None:
            self.betaG = 0.5
        
        # Instantiate sampler
        # Instantiate sampler 
        self.sampler = pm.Model(coords={"cluster" : range(n_components)}) 
        # Create pymc model 
        with self.sampler:
             self.mu = pm.Normal("mu",
                            mu = self.mu0, 
                            sigma = self.sigma0,
                            dims="cluster"
                            )
             self.sigma = pm.InverseGamma("sigma",
                                     beta = self.betaG,
                                     alpha = self.alphaG,
                                     dims = "cluster")
             
             self.weights = pm.Dirichlet("w", np.ones(n_components), dims="cluster")
    
    def fit(self, X):
        with self.sampler:
            obs = pm.NormalMixture("x",  w=self.weights,
                                   mu=self.mu, sigma=self.sigma, observed=X)
            mean_field = pm.fit(n = self.iterations,
                                method = "advi",
                                obj_optimizer=pm.adagrad_window(learning_rate=0.01))
            idata = mean_field.sample(self.n_samples)
        self.post = idata.posterior
    
    def sample_posterior(self):
        # Taking sample from one of the chains 
        sample_id = np.random.choice(range(self.n_samples), size=1)[0]
        sample = np.array([self.post["mu"].values[0, sample_id, :],
                           self.post["sigma"].values[0, sample_id, :],
                           self.post["w"].values[0, sample_id, :]])
        return sample
             
if __name__ == "__main__":
    import gym
    import numpy as np
    import ToyEnv
    import collections
    import itertools
    
    env = ToyEnv.ToyEnv()
    
    means = env._compute_means(0, 2)
    sigma = env.true_Sigma
    pis = env.true_pi
    data = np.zeros((100,))
    for i in range(100):
        k = np.random.choice(len(pis), p=pis)
        data[i] = np.random.normal(loc=means[k], scale=sigma[k])
        
    n_components = 3

    # Initialize prior parameters
    alpha = torch.ones(n_components)/n_components

    # Normal on mu
    mu0 = 0
    sigma0 = 1

    # Inverse gamma on sigma
    alphaG = 1
    betaG = 0.5 # scale paramterization
    
    # sampler = pm.Model(coords={"cluster" : range(n_components)}) 
    #     # Create pymc model 
    # with sampler:
    #     mu = pm.Normal("mu",
    #                     mu = mu0, 
    #                     sigma = sigma0,
    #                     dims="cluster"
    #                     )
    #     sigma = pm.InverseGamma("sigma",
    #                                 beta = 
    #                                 betaG,
    #                                 alpha = alphaG,
    #                                 dims = "cluster")
        
    #     weights = pm.Dirichlet("w", np.ones(n_components), dims="cluster")
            
    # with sampler:
    #     idata = pm.sample(2000, step = pm.HamiltonianMC())      
    
    # post = idata.posterior 
    # map_estimate = pm.find_MAP(model=sampler)

    # with sampler:
    #     pm.NormalMixture("x",  w=weights, mu=mu, sigma=sigma, observed=data)
        
    #     trace = pm.sample(2000, step = pm.HamiltonianMC(), return_inferencedata=False) 
    
    # hmc = HMCpymcGMM(1000, 3)
    
    # hmc.fit(data)
    # print(hmc.sample_posterior())
    
    # gibbs = GibbsSamplerGMM(1000, 3)
    # gibbs.fit(data)
    
    
    print("Fitting VI")
    vi = VIpymcGMM(1000, 3, iterations=100000)
    
    vi.fit(data)
    print(vi.post)
    print(vi.sample_posterior())
    # print(gibbs.sample_posterior())
    # print(type(samples))
    # print(type(samples.shape))
    # print(samples[np.random.choice(samples, size = 1)])
