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
    for i in range(m):
        maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(candidates_row, selected)
        
                
        s1array = rowsum[candidates]

        #print candidates_row
        #print candidates2 #0-9,21
        #print s1array
        #print K_subset.shape
            
        if len(selected) > 0:
            #print candidates.shape
            #print selected.shape
            #print candidates2.shape
            #print " Len(selected) >0"
            #print i 
            #print candidates2[selected]
            #print candidates2[candidates]
            temp = K_subset[candidates, :][:, candidates2[selected]]
            #print "info temp"
            #print temp
            #print temp.shape
            K_subset2 = K_subset[candidates,:][:,candidates2[candidates]]
            #print "info Ksubset2"
            #print K_subset2
            #print K_subset2.shape
            #print "np.sum(temp, axis=1)"
            #print np.sum(temp, axis=1)
            #print type(np.sum(temp, axis=1))
            #print "np.diagonal(K_subset2)"
            #print np.diagonal(K_subset2)
            #print type(np.diagonal(K_subset2))
            s2array = np.add(np.squeeze(np.asarray(np.sum(temp, axis=1))) *2, np.diagonal(K_subset2))
            #print "s2array"
            #print s2array
            #print s2array.shape
            #print "s2array type"
            #print type(s2array)
            s2array = np.squeeze(np.asarray(s2array/(len(selected) + 1)))
            #print "s2array info"
            #print s2array.shape
            #print s2array
            #print type(s2array) 
            #print "type s1array"
            #print type(s1array)
            s1array = np.subtract(np.squeeze(np.asarray(s1array)), s2array)
            #print "size s1array the earlier one"
            #print type(s1array)
            #print s1array.shape
            #print s1array

           
            
        
        else:
            K_subset2 = K_subset[candidates_row,:][:,candidates2]
            #print "k_subset2"
            #print K_subset2
            #print "s1"
            #print s1array
            #print "diagonal K_subset2"
            #print np.abs(np.diagonal(K_subset2))
            #print "new s1 shape"
            #print s1array.shape
            #print np.log(np.abs(np.diagonal(K_subset2))).shape
            #print "altern comp s1"
            #print np.log(np.abs(np.diagonal(K_subset2)))
            #print "Beginning now"
            #print np.squeeze(np.asarray(s1array)).shape
            #print np.log(np.abs(np.diagonal(K_subset2))).shape
            s1array = np.subtract(np.squeeze(np.asarray(s1array)), np.log(np.abs(np.diagonal(K_subset2))))
            #print " Len(selected) =0"
            #print s1array.shape
            #print type(s1array)
        
        print "we are out of the if-else"
        print "i"
        print i 
        print "s1 array"
        print s1array
        #print s1array.shape
        #print type(s1array)
        print "candidates"
        print candidates
        #print candidates.shape
        #print type(candidates2)
        #print np.argmax(s1array)
        argmax = candidates[np.argmax(s1array)]
        print "argmax"
        print argmax
        selected = np.append(selected, argmax)
        print "selected"
        print selected 
        #print selected.shape
        #print type(selected)
        print "candidates2"
        print candidates2 
        print candidates2.shape
        print type(candidates2)

        
    print candidates2[selected].shape
    return candidates2[selected]
