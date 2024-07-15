import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

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


def gen_data(theta, N, sigma):
    x = np.sort(10 * np.random.rand(N))
    # first evaluate the true model
    y = linear_model(theta, x)
    # add proportional noise
    y += np.abs(np.exp(theta[2]) * y) * np.random.randn(N)
    # add gaussian noise
    y += sigma * np.random.randn(N)
    data = np.stack([x, y], axis=1)
    return data


def likelihood(theta, data, sigma):
    """
    log of the likelihood function: P(data|theta)
    """
    obs = linear_model(theta, data[:, 0])
    adjusted_sigma2 = sigma**2 + obs**2 * np.exp(2 * theta[2])
    return -0.5 * np.sum(
        (obs - data[:, 1]) ** 2 / adjusted_sigma2 + np.log(adjusted_sigma2)
    )


def prior(theta):
    """
    log of prior probability distribution: P(theta)
    """
    if (-5 < theta[0] < 5) and (0 < theta[1] < 10) and (-10 < theta[2] < 1):
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
theta_true = np.array([-0.5, 5, np.log(f)])
print("True values: ", theta_true)
sigma_meas = 0.2
N = 50
data = gen_data(theta_true, N, sigma_meas)


# Proxy likelihood function for MLE using scipy
def cost_function(*args):
    return -likelihood(*args)


theta0 = np.array([-0.4, 4.8, np.log(0.1)])  # random initial guess
MLE = sp.optimize.minimize(cost_function, theta0, args=(data, sigma_meas))
theta_mle = MLE.x
print("MLE : ", theta_mle)

# Setup the posterior object
post = Posterior(data, sigma_meas)

# Initialize the number of chains
n_walkers = 32
n_dim = len(theta_true)
theta0 = theta_mle + 0.4 * np.random.rand(n_walkers, n_dim)
# Run the MCMC to get posterior samples
sampler = emcee.EnsembleSampler(n_walkers, n_dim, post)
sampler.run_mcmc(theta0, 10000, progress=True)
# Flatten the chains, and thin them out
flat_samples = sampler.get_chain(discard=1000, thin=200, flat=True)

# Plot the posterior
fig = corner.corner(flat_samples, labels=["$m$", "$b$", "$\log(f)$"], truths=theta_true)
fig.savefig("posterior_joint.pdf")

## Data plots and posterior predictive checks
plt.figure()
rand_sample_idx = np.random.randint(len(flat_samples), size=100)
label_set = False
for i in rand_sample_idx:
    theta_s = flat_samples[i]
    if label_set:
        plt.plot(
            data[:, 0],
            linear_model(theta_s, data[:, 0]),
            alpha=0.1,
            color="grey",
            ls="solid",
        )
    else:
        label_set = True
        plt.plot(
            data[:, 0],
            linear_model(theta_s, data[:, 0]),
            alpha=0.1,
            color="grey",
            label="posterior",
            ls="solid",
        )
plt.errorbar(
    data[:, 0],
    data[:, 1],
    ls="none",
    marker="o",
    yerr=sigma_meas,
    capsize=0,
    label="noisy data",
    ms=2,
)
plt.plot(data[:, 0], linear_model(theta_true, data[:, 0]), ls="dotted", label="truth")
plt.plot(
    data[:, 0],
    linear_model(theta_mle, data[:, 0]),
    ls="dashed",
    label="MLE",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend(loc="best")
plt.savefig("posterior_predictive.pdf")
