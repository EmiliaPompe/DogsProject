import numpy as np
import scipy

def rbf(x,y, gamma):
    x = np.asarray(x)
    y = np.asarray(y)
    return scipy.exp(-gamma*np.linalg.norm(x-y))

def our_rbf_kernel(current_kernel_subset,  gamma, x_new, n, timeseries):
    new_row = numpy.zeros(shape=(1,n))
    for i in range(n):
         new_row[i] = rbf(x_new, timseries[i])
    new_kernel_subset = np.vstack(current_kernel_subset, new_row)
    return new_kernel_subset

def compute_kernel(data, N, gamma):
    kernel_matrix = np.zeros((N, N))
    for I in range(N):
        for J in range(N):
            kernel_matrix[I,J] = rbf(x = data[I,:], y = data[J,:], gamma = gamma)
    
    return kernel_matrix