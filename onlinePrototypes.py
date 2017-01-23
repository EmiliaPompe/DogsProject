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
import sys

def protonline(K_subset, selectedprotos, m):

    n = np.shape(K_subset)[1]  #number of columns in the rectangular matrix    
    rowsum = 2*np.sum(K_subset, axis=1) / n

    candidates2 = np.append(selectedprotos, n-1)  # indecides of the previous prototypes group + the new observation
    candidates_row = range(m+1)    
    
    
    selected = np.array([], dtype=int) #vector to store the selected prototypes
    value = np.array([])
    for i in range(m+1):
        maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(candidates_row, selected)
        
                
        s1array = rowsum[candidates]

        print candidates_row
        print candidates2 #0-9,21
        print s1array
        print K_subset.shape
            
        if len(selected) > 0:
            print candidates.shape
            print selected.shape
            print candidates2.shape
            temp = K_subset[candidates, candidates2[selected]]
            K_subset2 = K_subset[candidates,candidates2[candidates]]
            print K_subset2.shape
            s2array = np.sum(temp, axis=1) *2 + np.diagonal(K_subset2)
            s2array = s2array/(len(selected) + 1)
            s1array = s1array - s2array
        
        else:
            K_subset2 = K_subset[candidates_row,:][:,candidates2]
            print "k_subset2"
            print K_subset2
            print "s1"
            print s1array
            print "diagonal K_subset2"
            print np.abs(np.diagonal(K_subset2))
            s1array = s1array - np.log(np.abs(np.diagonal(K_subset2)))
            print "new s1"
            print s1array
            print "altern comp s1"
            print np.log(np.abs(np.diagonal(K_subset2)))
            s1array = np.subtract(s1array,np.asmatrix(np.log(np.abs(np.diagonal(K_subset2)))))
            print type(s1array)
            print type(np.log(np.abs(np.diagonal(K_subset2))))
            print K_subset2.shape
            print s1array.shape
        
        print "hello"
        print np.argmax(s1array)
        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)

    return candidates2[selected]