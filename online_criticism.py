##############################################################################################################################
# Function to select criticisms
# K_subset is a subset of the kernel matrix of dimensions equal to  pontential criticisms (c+1) x (number of data so far) 
# selectedprotos: prototypes already selected
# selectedcritic: criticism selected at each step
# c : number of criticisms to be selected
# reg: regularizer type.
# RETURNS: indices selected as criticisms
# selected prototype is a vector from 0:m-1 coming from the prototype function
# n = ncols of the kernel matrix in our case is (n) cause you never have the kernel matrix computed for 
# the complete dataset
# candidates2 is a vector with the indices for the criticism expressed with respect to the overall dataset
# candidates_row is othe set of indices wrt to row of the K_subset
##############################################################################################################################

import numpy as np
#import sys


def greedy_select_criticism_online(K_subset, selectedprotos, selectedcritic, c, reg='logdet'):

    n = np.shape(K_subset)[1]  #number of columns in the rectangular matrix          
    if reg in ['None','logdet']:
        pass
    else:
        print "wrong regularizer :" + regularizer
        exit(1)
    options = dict()

    selected = np.array([], dtype=int)            # vector in which we will store the criticism
    candidates2 = np.append(selectedcritic, n-1)  # indecides of the previous criticism group + the new observation
    candidates_row = range(c+1)
    inverse_of_prev_selected = None  # should be a matrix

    rowsum = np.sum(K_subset, axis=1)/n

    for i in range(c+1):
        #maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(candidates_row, selected) #wrt to row of the K_subset

        s1array = rowsum[candidates]

        temp = K_subset[:,selectedprotos]
        s2array = np.sum(temp, axis=1)

        s2array = s2array / (len(selectedprotos))

        s1array = np.abs(s1array - s2array)
        if reg == 'logdet':
            if inverse_of_prev_selected is not None: # first call has been made already
                temp = K_subset[candidates, :][:, candidates2[selected]]
                K_subset2 = K_subset[candidates,:][:,candidates2[candidates]]
                
                temp2 = np.array(np.dot(inverse_of_prev_selected, temp.transpose()))
                regularizer = temp2 * temp.transpose()
                regcolsum = np.sum(regularizer, axis=0)
                regularizer = np.log(np.abs(np.diagonal(K_subset2) - regcolsum))
                s1array = s1array + regularizer
            else:
                K_subset2 = K_subset[candidates_row,:][:,candidates2]
                s1array = s1array - np.log(np.abs(np.diagonal(K_subset2)))
                    
        argmax = candidates[np.argmax(s1array)]
        maxx = np.max(s1array)

        selected = np.append(selected, argmax)
        if reg == 'logdet':
            KK = K_subset[selected,:][:,candidates2[selected]]
            
            inverse_of_prev_selected = np.linalg.inv(KK) # shortcut

    return candidates2[selected]
