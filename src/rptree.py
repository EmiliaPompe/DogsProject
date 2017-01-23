import numpy as np

def random_unit_vector(D, C=1):
    p = np.random.randn(D, C)
    return p / np.sqrt((p**2).sum(0))



class RPTree(object):
    """ Implementation of a random projection tree to be used for nearest neighbor 
        search [1].
        
        [1] Dasgupta, S. and Sinha, K. (2015). Randomized partition trees for nearest 
            neighbor search. _Algorithmica_, 72(1):237--263. """

    def __init__(self, inputs, target_leaf_size=None, copy=True):
        self.M, self.D = inputs.shape
        self._entries = np.array(inputs, copy=copy)
        target_leaf_size = target_leaf_size or int(2*np.log2(self.M))
        self.tree_depth = int(np.ceil(np.log2(float(self.M) / target_leaf_size)))
        self._rebalance_thresh = target_leaf_size+1
        self._inserts = 0
        print "Number of items:", self.M
        print "Each item dimension:", self.D
        print "Tree depth:", self.tree_depth
        print "Expected number of points at leaf:", (self.M / 2.0**self.tree_depth)
        self._proj = random_unit_vector(self.D, (2**self.tree_depth-1))
        self.reset_split_points()
        
    def reset_split_points(self):
        ''' this is costly: O(M \log M). Don't do it too often. '''
        self._split = np.empty((2**self.tree_depth-1))
        self._split_subtree(np.arange(self.M), 1)
        
        self._leaf_nodes = np.empty((2**self.tree_depth,), dtype=object)
        for e in xrange(2**self.tree_depth):
            self._leaf_nodes[e] = set({})
        for j in xrange(self.M):
            this_bin = self._get_bin(self._entries[j])
            self._leaf_nodes[this_bin].add(j)

    def __getitem__(self, ix):
        return np.array(self._entries[ix])
        
    def __setitem__(self, ix, value):
        self._inserts += 1
        incremental = True
        if self._inserts >= self.M:
            # consider rebuilding the split points and leaf index
            self._inserts = 0
            incremental = np.max(map(len, self._leaf_nodes)) <= self._rebalance_thresh
        if incremental:
            old_bin = self._get_bin(self._entries[ix])
            new_bin = self._get_bin(value)
            self._entries[ix] = value
            self._leaf_nodes[old_bin].remove(ix)
            self._leaf_nodes[new_bin].add(ix)
            # TODO: in the future, we could consider automatically rebalancing the tree
            #       here, if it is now too unbalanced
        else:
            self._entries[ix] = value
            self.reset_split_points()
    
    def _split_subtree(self, nodes, ix):
        if ix-1 < len(self._split):
            outcomes = np.dot(self._proj[:,ix-1].T, self._entries[nodes].T)
            beta = np.random.rand()*50+25
            self._split[ix-1] = np.percentile(outcomes, beta)
            self._split_subtree(nodes[outcomes <= self._split[ix-1]], 2*ix) # left
            self._split_subtree(nodes[outcomes > self._split[ix-1]], 2*ix+1) # right

    def _get_bin(self, phi):
        """ Find the correct bin index for a given input vector `phi` """
        ix = 1
        for d in xrange(self.tree_depth):
            direction = np.dot(self._proj[:,ix-1].T, phi) > self._split[ix-1]
            ix = 2*ix + direction
        return ix - 2**self.tree_depth
        
    def search(self, target):
        leaf = self._get_bin(target)
        searchlist = set(self._leaf_nodes[leaf])
        if len(searchlist) == 0:
            # Empty bin --- consulting siblings
            sibling = leaf + 1 - 2*np.mod(leaf,2)
            searchlist.update(self._leaf_nodes[sibling])
        return searchlist
