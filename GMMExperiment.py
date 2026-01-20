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
        The updated weights of the Gaussian Mixture Model.
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

def gmm_distance(GMM_1_weights: np.ndarray, GMM_2_weights: np.ndarray
                 ) -> np.floating:
    """Compute the distance between two GMMs to determine nearest neighbors

    Since we only update the mixture weights, we consider the distance between
    two Gaussian Mixture Models to be the Euclidean distance between their 
    weight vectors.

    Args:
        GMM_1_weights: A 1-D numpy array representing the mixture weights for
            the first GMM.
        GMM_2_weights: A 1-D numpy array representing the mixture weights for
            the second GMM.

    Returns:
        A nonnegative float representing the Euclidean distance between the
        weight vectors. 
    """
    return np.linalg.norm(GMM_1_weights-GMM_2_weights, ord='fro')

def generate_distance_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    """Compute the NxN matrix of pairwise distances between N GMMs with weights
    specified by the weight matrix.

    Args:
        weight_matrix: An Nxd matrix containing the weights of N GMMs with d
            components each. `weight_matrix[i][j]` specifies the weight of the 
            jth component in the ith GMM.

    Returns:
        A symmetric NxN matrix where `distance_matrix[i][j]` specifies the 
        distance between the ith and jth GMM.  
    """
    num_GMMs, _ = weight_matrix.shape
    distance_matrix = np.zeros((num_GMMs, num_GMMs))
    for i in range(num_GMMs):
        for j in range(i+1, num_GMMs):
            distance_matrix[i][j] = gmm_distance(weight_matrix[i, :], 
                                                 weight_matrix[j, :])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix

def experiment(time_steps: int, mirror_probability: float,
               num_nearest_neighbors: int, RAG_size: int, 
               initial_gmm_weights: np.ndarray, gmm_means: np.ndarray, 
               gmm_stddev: np.ndarray, seed: int | None = None
               ) -> list[np.ndarray]:
    """Conduct an instance of the GMM experiment as described in the paper.

    Args:
        time_steps: A nonnegative integer representing the number of time steps
            to run the simulation for.
        mirror_probability: A float between 0 and 1 representing the probability
            an agent will "mirror" during its interaction step and query itself.
        num_nearest_neighbors: An integer between 1 and the number of agents-1
            representing the number of nearest neighbors to consider when
            conducting an interaction step without mirroring
        RAG_size: A positive integer representing the number of elements in the
            RAG set. 
        initial_gmm_weights: A matrix of size Nxd consisting of the initial
            weights for the agents in the experiment. 
            `initial_gmm_weights[i][j]` represents the initial weight of the jth
            component of the ith GMM.
        gmm_means: A matrix of size d consisting of the fixed means for each 
            Gaussian component for all GMMs.
        gmm_stddev: A matrix of size d consisting of the fixed standard
            standard deviations for each Gaussian component for all GMMs.
        seed: An optional parameter allowing the seed for the random number
            generator to be set.
    """
    if seed is not None:
        _rng = np.random.default_rng(seed)

    num_agents, _ = initial_gmm_weights.shape


    gmm_weights_history = [initial_gmm_weights]

    return gmm_weights_history
