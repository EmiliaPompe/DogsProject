import numpy as np
from scipy import optimize, stats



class Herder(object):
    """ Implementation of kernel herding [1], using an explicit feature space. 
        This is used as a benchmark, and as a sanity check.
        
        [1] Chen, Y., Welling, M., and Smola, A. (2010). Super-samples from kernel herding. 
            In _Proceedings of the Twenty-Sixth Conference on Uncertainty in Artificial 
            Intelligence_ (UAI2010). """

    def __init__(self, feature_fn, sampler_fn, target_features, budget=100, run_optimizer=True):
        """ The herder requires a function `feature_fn` which computes the features from 
            the data, a function `sampler_fn` which samples a new candidate point from a
            proposal distribution, and a vector `target_features` representing the kernel
            mean embedding of the target distribution.
            
            Optimization procedes by considering a fixed `budget` number of candidate
            points, then (optionally) runs an optimization routine to further refine the
            best candidate point. """
        self._compute_phi = feature_fn
        self._target_phi = target_features
        self._draw_sample = sampler_fn
        self._run_optimizer = run_optimizer
        self._points = []
        self.N = 0
        self._herded_phi = None
        self._budget = budget

    def get_samples(self, N, verbose=False):
        if self.N < N:
            for n in xrange(N-self.N):
                if verbose:
                    print "Herding sample #%d" % (self.N + 1)
                self._herd_new_sample()
        return np.array(self._points[:N])
    
    def _objective(self, x):
        phi_x = self._compute_phi(np.array([x])).mean(0)
        obj = np.dot(phi_x, self._target_phi)
        if self.N > 0:
            obj -= np.dot(phi_x, self.N*self._herded_phi)/(self.N+1)
        return obj
        
    def _search(self):
        score = -np.Inf
        best = None
        for i in xrange(self._budget):
            x = self._draw_sample()
            new_score = self._objective(x)
            if new_score > score:
                score = new_score
                best = x
        if self._run_optimizer:
            new_best = optimize.minimize(lambda x: -self._objective(x), best)['x']
            assert self._objective(new_best) >= self._objective(best)
            return new_best
        else:
            return best
        
    def _herd_new_sample(self):
        next_value = self._search()        
        self._points.append(next_value)
        next_phi = self._compute_phi(np.array([next_value])).mean(0)
        if self.N == 0:
            self._herded_phi = next_phi
        else:
            self._herded_phi = (self.N*self._herded_phi + next_phi)/(self.N+1)
        self.N += 1
