import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def median_heuristic(points, percentile=50, max_points=200):
    """ Implementation of median heuristic to choose kernel lengthscale """
    assert len(points) < max_points
    dim = points[0].shape[0]
    pairwise = pairwise_distances(points)
    pairwise = pairwise[np.tri(len(pairwise), k=-1, dtype=bool)].ravel()
    return (np.percentile(pairwise, percentile)*1.0)**2


class RFF(object):
    """ Random Fourier Features implementation, Gaussian kernel """

    def __init__(self, num_features, dim, lengthscale=1.0):
        self.num_features = num_features
        self.omega = np.random.randn(num_features,dim)/np.sqrt(lengthscale)
        self.b = np.random.uniform(0.0, 2*np.pi, size=(num_features,1))
        self._const = np.sqrt(2.0/self.num_features)

    def __call__(self, X):
        return (np.cos(np.dot(X, self.omega.T) +  self.b.T)*self._const)