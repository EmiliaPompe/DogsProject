import numpy as np
import matplotlib.pyplot as plt
import sys
import time
#sys.path.insert(0, '../src')
from scipy import stats

class SyntheticTarget(object):
    
    def __init__(self, dist_components=2, dim=2, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.dist_weights = stats.dirichlet(np.ones(dist_components)*10).rvs().ravel()
        dist_centers = np.random.uniform(-4, 4, size=(dist_components, dim))

        def gen_cov():
            assert(dim == 2)
            rho = np.random.rand()-0.5
            sd = np.random.rand(dim)*0.4+0.5
            L = np.zeros((dim,dim))
            L[0,0] = sd[0]**2
            L[1,1] = sd[1]**2
            L[0,1] = rho*sd.prod()
            return np.dot(L, L.T)

        dist_cov = np.array([gen_cov() for i in xrange(dist_components)])
        self.dist_obj = np.array([stats.multivariate_normal(dist_centers[k], dist_cov[k]) for k in xrange(dist_components)], dtype=object)
        self.dim = dim

    def compute_pdf(self, x):
        return np.sum(self.dist_weights[:,None,None] * np.array([d.pdf(x) for d in self.dist_obj]), 0)

    def draw_sample(self, ns=1):
        component = np.random.multinomial(ns, self.dist_weights) # np.argmin(np.random.rand() > self.dist_weights.cumsum())
        offset = np.concatenate(([0], component.cumsum()[:-1]))
        samples = np.empty((ns,self.dim))
        for c, n in enumerate(component):
            if n > 0:
                samples[offset[c]:offset[c]+n] = self.dist_obj[c].rvs(n) 
        np.random.shuffle(samples)
        return samples

    def draw_sample_slow(self):
        component = np.argmin(np.random.rand() > self.dist_weights.cumsum())
        return self.dist_obj[component].rvs()

    def plot(self):
        delta = 0.1
        boundary = 5
        x = np.arange(-boundary, boundary, delta)
        y = np.arange(-boundary, boundary, delta)
        X, Y = np.meshgrid(x, y)
        Z = np.log(self.compute_pdf(np.concatenate((X[:,:,None],Y[:,:,None]),2)))
        plt.figure()
        contour_levels = np.arange(-9,1)
        CS = plt.contour(X, Y, Z, levels=contour_levels)
        plt.set_cmap('Blues')
        
        
# Change for different sampled target dist
SEED = 666
ex = SyntheticTarget(seed=SEED, dist_components=2)

sample_size = 1000
M = 100
stream = ex.draw_sample(sample_size)
#stream.shape finds the dimensions of the nparray stream
print stream.shape, "=>", M




