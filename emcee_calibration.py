import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import run_rans_nofcs as rr

plt.style.use("project.mplstyle")
np.random.seed(42)

"""
Bayesian Calibration framework:
    Given data with uncertainty, compute the posterior probability distribution of parameters theta using Bayes rule.
    p(theta|d)  \propto  p(d|theta)   p(theta)
    ^ posterior          ^likelihood  ^prior
"""


def linear_model(theta, x):
    """
    simple linear model: y = mx + b
    """
    return theta[0] * x + theta[1]


def gen_data(theta, sigma):
    wake_stats = rr.run_rans(theta)
    nx, nf = wake_stats.shape
    noise = np.random.randn(nx, nf - 2) * sigma
    noisy_stats = wake_stats.copy()
    noisy_stats[:, 1:-1] += noise
    return noisy_stats


def likelihood(theta, data, sigma):
    """
    log of the likelihood function: P(data|theta)
    """
    try:
        obs = rr.run_rans(theta)
        llhood = -0.5 * np.sum((obs - data) ** 2) / sigma**2
    except Exception:
        llhood = -np.inf
    return llhood


def prior(theta):
    """
    log of prior probability distribution: P(theta)
    """
    if (0.001 < theta[0] < 0.1) and (0.1 < theta[1] < 10) and (0.1 < theta[2] < 10):
        return 0
    return -np.inf


class Posterior:
    """
    Stores the data object and computes the posterior probability
    """

    def __init__(self, data, sigma):
        self.data = data
        self.sigma = sigma

    def __call__(self, theta):
        return likelihood(theta, self.data, self.sigma) + prior(theta)


# Set true parameters and generate data
f = 0.5
theta_true = np.array([0.02, 1.0, 2.0])
print("True values: ", theta_true)
sigma_meas = 0.01
data = gen_data(theta_true, sigma_meas)


# Proxy likelihood function for MLE using scipy
def cost_function(*args):
    return -likelihood(*args)


theta0 = [5.48 ** (-2), 1.176, 1.92]
# MLE = sp.optimize.minimize(cost_function, theta0, args=(data, sigma_meas))
# theta_mle = MLE.x
# print("MLE : ", theta_mle)

# Setup the posterior object
post = Posterior(data, sigma_meas)

# Initialize the number of chains
n_walkers = 6
n_dim = len(theta_true)
theta0 = theta0 + 0.01 * np.random.rand(n_walkers, n_dim)
# Run the MCMC to get posterior samples
sampler = emcee.EnsembleSampler(n_walkers, n_dim, post)
sampler.run_mcmc(theta0, 1000, progress=True)
# Flatten the chains, and thin them out
flat_samples = sampler.get_chain(discard=0, thin=10, flat=True)

fig_name = "posterior.pdf"
with PdfPages(fig_name) as pdf:
    # Plot the posterior
    fig = corner.corner(
        flat_samples, labels=[r"$C_{\mu}$", r"$C_{1e}$", r"$C_{2e}$"], truths=theta_true
    )
    pdf.savefig(figure=fig)
