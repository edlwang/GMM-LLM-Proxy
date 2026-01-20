# Imports
import numpy as np
from scipy.stats import norm

# Global Random Number Generator
_rng = np.random.default_rng(1)

# Functions
def sample_GMM(mixture_weights: np.ndarray, mixture_means: np.ndarray, 
               mixture_stddev: np.ndarray, num_samples: int) -> np.ndarray:
    """Sample points from a Gaussian Mixture Model (GMM).

    Given a GMM specified by the weights on each component and the mean and 
    standard deviations associated with each Gaussian component, sample a 
    specified number of points from the GMM. 
    
    Args:
        mixture_weights: A 1-D numpy array of nonnegative real numbers that sums
            to 1, representing the weights for each Gaussian component. The kth
            entry `mixture_weights[k]` is the weight associated with the kth 
            component.
        mixture_means: A 1-D numpy array of real numbers representing the mean
            of each Gaussian component. The kth entry `mixture_means[k]` is the
            mean of the kth Gaussian component.
        mixture_stddev: A 1-D numpy array of nonnegative real numbers
            representing the standard deviation of each Gaussian component. The
            kth entry `mixture_stddev[k]` is the standard deviation of the kth
            Gaussian component.
        num_samples: A positive integer representing the number of samples to
            generate.

    Returns:
        A numpy array with `num_samples` independent and identically distributed
        samples from the specified GMM distribution.
    """
    num_components = len(mixture_weights)
    samples = np.zeros(num_samples)
    for idx in range(num_components):
        # First sample all the Gaussian components
        component = _rng.choice(a=num_components, size=num_samples, 
                                p=mixture_weights)
        # Next sample from the normal distribution defined by the Gaussian
        samples[idx] = _rng.normal(loc=mixture_means[component], 
                                   scale=mixture_stddev[component])
    return samples

def update_GMM(data: np.ndarray, mixture_weights: np.ndarray, 
               mixture_means: np.ndarray, mixture_stddev: np.ndarray
               , epsilon: float = 1e-12) -> np.ndarray:
    """Update the weights of the Gaussian Mixture Model (GMM) via EM algorithm

    Given a GMM specified by the weights on each component and the mean and
    standard deviations associated with each Gaussian component, return the new
    weights after one step of the EM algorithm.  

    Args:
        data: A 1-D numpy array of real numbers representing the new data used
            to update the GMM. 
        mixture_weights: A 1-D numpy array of nonnegative real numbers that sums
            to 1, representing the initial weights for each Gaussian component. 
            The kth entry `mixture_weights[k]` is the weight associated with the 
            kth component.
        mixture_means: A 1-D numpy array of real numbers representing the 
            initial mean of each Gaussian component. The kth entry 
            `mixture_means[k]` is the mean of the kth Gaussian component.
        mixture_stddev: A 1-D numpy array of nonnegative real numbers
            representing the initial standard deviation of each Gaussian 
            component. The kth entry `mixture_stddev[k]` is the standard 
            deviation of the kth Gaussian component.
        epsilon: A small positive real number to truncate zero values to so 
            operations are well-conditioned. 

    Returns:
        The updated weights of the Gaussian Mixture Model
    """
    num_components = len(mixture_weights)
    num_data_points = len(data)

    updated_weights = np.copy(mixture_weights)
    updated_means = np.copy(mixture_means)
    updated_stddev = np.copy(mixture_stddev)
    # Truncate to epsilon
    zero_indices = np.where(mixture_weights==0)
    updated_weights[zero_indices] = epsilon

    # Compute updated weights using EM 
    data_posterior = np.zeros((num_components, num_data_points))
    for idx, data_point in enumerate(data):
        # compute likelihood of drawing the data point from each component
        data_posterior[:, idx] = updated_weights * norm.pdf(
            data_point, updated_means, updated_stddev)
        # normalize
        data_posterior[:, idx] /= np.sum(data_posterior[:, idx])
    updated_weights = np.sum(data_posterior, axis=1).clip(
        max=1/epsilon, min=epsilon)
    updated_weights[np.isnan(updated_weights)] = epsilon
    updated_weights /= np.sum(updated_weights)
    return updated_weights
