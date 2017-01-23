##############################################################################################################################
# Function to select prototypes
# K_subset is a subset of the kernel matrix of dimensions equal to  pontential prototypes (m+1) x (number of data so far) 
# selectedprotos: prototypes already selected
# m : number of prototypes to be selected
# is_K_sparse:  True means K is the pre-computed  csc sparse matrix? False means it is a dense matrix.
# RETURNS: indices selected as prototypes
# n = number of cols of the kernel matrix in our case is (n) cause you never have the kernel matrix computed for 
# the complete dataset
# candidates2 is a vector with the indices for the criticism expressed with respect to the overall dataset
# candidates_row is othe set of indices wrt to row of the K_subset
#############################################################################################################################

import numpy as np
#import sys

def greedy_select_protos_online(K_subset, selectedprotos, m):

    n = np.shape(K_subset)[1]  #number of columns in the rectangular matrix    

    rowsum = 2*np.sum(K_subset, axis=1) / n

    candidates2 = np.append(selectedprotos, n)  # indecides of the previous prototypes group + the new observation
    candidates_row = range(m+1)    
        
    selected = np.array([], dtype=int) #vector to store the selected prototypes
    value = np.array([])
    for i in range(m+1):
        #maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(candidates_row, selected)

        s1array = rowsum[candidates]
        
        print s1array.shape
        
        if len(selected) > 0:
            temp = K_subset[candidates, :][:, candidates2[selected]]
            K_subset2 = K_subset[candidates,:][:,candidates2[candidates]]
            s2array = np.sum(temp, axis=1) *2 + np.diagonal(K_subset2)
            s2array = s2array/(len(selected) + 1)
            s1array = s1array - s2array
        print s1array.shape
        
        else:
            K_subset2 = K_subset[candidates_row,:][:,candidates2]
            s1array = s1array - np.log(np.abs(np.diagonal(K_subset2)))
        
        print s1array.shape 
        
        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)

    return candidates2[selected]