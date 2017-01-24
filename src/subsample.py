import numpy as np
from scipy import optimize, stats

import rptree

def log_sum_exp(a, b):
    A = max(a, b)
    return A + np.log(np.exp(a - A) + np.exp(b - A))


class Subsampler(object):
    ''' Base class for all stream-subsamplers '''
    #Points: set of data that we are looking at.
    #Phi, phi_hat from the paper: the vector of random fourier features (as 
    
    
    def __init__(self, points, phi, initial_log_normalizer=None, logging=True, ground_truth_phi=None, test_fns=None):
        self.points = np.array(points)  #Puts the points in an np array
        self._compute_phi = phi         #Sets the compute_phi attribute to phi_hat the function!
        self._each_phi = phi(points)     #phi_hat computed on the prototypes ie the first M
        self._full_mean_phi = self._each_phi.mean(0)            #mu_hat_n
        
        self.total_sum_test = np.sum(self._each_phi, axis = 0)
        self.sample_sum_test = np.sum(self._each_phi, axis = 0)
        
        self._subset_mean_phi = self._each_phi.mean(0)           #new_hat^n_M
        self.M = len(points)                #Initially M=N as you have the same number of prototypes as data in total
        self.N = len(points)
        
        #The below means that it doesn't normalize things and so computes the usual sum.
        self._log_normalizer = initial_log_normalizer if initial_log_normalizer is not None else np.log(self.M)
        self.which = np.arange(self.M)
        
        # logging
        self._ground_truth_phi = ground_truth_phi
        self._logging = logging
        self._test_fns = test_fns
        if logging:
            self.accepted = []
            self.MMD_online = []
            if test_fns is not None:
                self.fn_output_sub = []
                self.fn_output_all = []
                for i in xrange(len(test_fns)):
                    self.fn_output_sub.append([])
                    self.fn_output_all.append([])
            if ground_truth_phi is not None:
                self.MMD_sub_gt = []
                self.MMD_all_gt = []
    
    def consider(self, candidate, log_weight=0.0):
        next_phi = self._compute_phi(np.array([candidate])).mean(0) 
        #note .mean(0) doesn't do anything here, as it calculates the column mean of a vector so leaves it as a vector.
        #This can be tested eg using:
        #print self._compute_phi(np.array([candidate]))
        #print 'inbetween'
        #print next_phi
        
        #Note, the default is that log_weight = 0, in which case this is just the usual mean
        updated_normalizer = log_sum_exp(self._log_normalizer, log_weight)
        weight_next = np.exp(log_weight - updated_normalizer)
        weight_prev = np.exp(self._log_normalizer - updated_normalizer)
        
        self.total_sum_test += next_phi
        
        self._full_mean_phi[:] = weight_next*next_phi + weight_prev*self._full_mean_phi    #Eqn (19) from notes
        self._log_normalizer = updated_normalizer
        self.N += 1
        # run nearest neighbor search to find which feature to drop!
        target_phi = next_phi + self.M*(self._subset_mean_phi - self._full_mean_phi)
        best_ix = self._nn_search(target_phi, next_phi)
        # now, maybe swap
        accepted = best_ix is not None
        if accepted:
            self._subset_mean_phi += (next_phi - self._each_phi[best_ix])/self.M     #Eq (21)
            
            self.sample_sum_test += next_phi - self._each_phi[best_ix]
            
            self._each_phi[best_ix] = next_phi
            self.points[best_ix] = candidate
            self.which[best_ix] = self.N-1
        if self._logging:
            self._log_status(np.array([candidate]), accepted)
    
    def _log_status(self, candidate, accepted):
        self.accepted.append(accepted)
        subsampled_norm_squared = np.sum(self._subset_mean_phi**2,0)
        all_norm_squared = np.sum(self._full_mean_phi**2,0)
        self.MMD_online.append(np.sqrt(all_norm_squared + subsampled_norm_squared - 2*np.dot(self._subset_mean_phi, self._full_mean_phi)))
        # compare against an external "ground truth" E[phi]:
        if self._ground_truth_phi is not None:
            gt_norm = np.sum(self._ground_truth_phi**2,0)
            self.MMD_sub_gt.append(np.sqrt(gt_norm + subsampled_norm_squared - 2*np.dot(self._subset_mean_phi, self._ground_truth_phi)))
            self.MMD_all_gt.append(np.sqrt(gt_norm + all_norm_squared - 2*np.dot(self._full_mean_phi, self._ground_truth_phi)))
        # compute expectations of various test functions
        if self._test_fns is not None:
            for i, fn in enumerate(self._test_fns):
                if accepted or len(self.fn_output_sub[i]) == 0:
                    # only re-compute if we updated the point set
                    self.fn_output_sub[i].append(np.mean(fn(self.points), 0))
                else:
                    self.fn_output_sub[i].append(np.array(self.fn_output_sub[i][-1]))
                if len(self.fn_output_all[i]) == 0:
                    last_avg = self.fn_output_sub[i][-1]
                else:
                    last_avg = self.fn_output_all[i][-1]
                self.fn_output_all[i].append((fn(candidate)[0] + (self.N - 1)*last_avg)/self.N)
            
    def _nn_search(self, target_phi, candidate_phi):
        raise NotImplementedError("")


class LinearSubsampler(Subsampler):
    """ Linear-time subsampler: nearest neighbor search through linear scan of all
        current points. """
    
    def _nn_search(self, target_phi, candidate_phi):
        feature_err = np.empty((self.M,))
        for j in range(self.M):
            feature_err[j] = ((target_phi - self._each_phi[j])**2).sum()
        baseline = ((target_phi - candidate_phi)**2).sum()
        best_ix = np.argmin(feature_err)
        if feature_err[best_ix] > baseline:
            best_ix = None
        return best_ix


class TreeSubsampler(Subsampler):
    """ Logarithmic-time subsampler: approximate nearest-neighbor search using random 
        projection tree. """
    
    def __init__(self, points, phi, benchmark=False, **kwargs):
        """ If `benchmark` is true, then this will additionally run an exact NN search,
            and compare the selections of the random projection tree to the actual nearest
            neighbors. This will drastically affect the runtime, and is only used to 
            evaluate the quality of the approximate search. """
        super(self.__class__, self).__init__(points, phi, **kwargs)
        print "Size of subsample:", self.M
        self.run_benchmarks = benchmark
        D = len(self._subset_mean_phi)
        if self._logging:
            self.counts = []
        if benchmark:
            self.could_have_accepted = []
            self.found_best = []
        # build tree
        self._rptree = rptree.RPTree(self._each_phi)


    def _nn_search(self, target_phi, candidate_phi):

        baseline = ((target_phi - candidate_phi)**2).sum()
        best_ix = None

        searchlist = self._rptree.search(target_phi)

        for ix in searchlist:
            this_err = ((target_phi - self._each_phi[ix])**2).sum()
#             print ix, "->", this_err # useful for debugging
            if this_err < baseline:
                baseline = this_err
                best_ix = ix

#         print "best found:", baseline, "ix", best_ix, "at location", best_loc
        if self._logging:
            self.counts.append(len(searchlist))
        if self.run_benchmarks:
            all_feature_err = np.empty((self.M,))
            for j in xrange(self.M):
                all_feature_err[j] = ((target_phi - self._each_phi[j])**2).sum()
            full_best_ix = np.argmin(all_feature_err)
#             print "do nothing:", ((target_phi - candidate_phi)**2).sum()
#             print "best alternative:", full_best_ix, "->", all_feature_err[full_best_ix]
            if all_feature_err[full_best_ix] > ((target_phi - candidate_phi)**2).sum():
                full_best_ix = None
            self.found_best.append(best_ix == full_best_ix)
            self.could_have_accepted.append(full_best_ix is not None)

#             print "compare:", best_ix, full_best_ix

        if best_ix is not None:
            self._rptree[best_ix] = candidate_phi
            
        return best_ix
