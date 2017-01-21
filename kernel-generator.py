from sklearn.metrics.pairwise import rbf_kernel

   	def calculate_kernel(self, g=None):
        if g is None:
            if self.gamma is None:
                print "gamma not provided!"
                exit(1)
            else:
                self.kernel = rbf_kernel(self.X, gamma=self.gamma)
        else:
            self.kernel = rbf_kernel(self.X, gamma=g)


