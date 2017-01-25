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
import sys


def crionline(K_subset, selectedprotos, selectedcritic, c, reg='logdet'):

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

    for i in range(c):
        
        maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(candidates_row, selected) #wrt to row of the K_subset
        
        #print  "s1array"   
        s1array = np.squeeze(np.asarray(rowsum[candidates]))
        #print s1array.shape
        #print type(s1array)

        temp = K_subset[candidates, :][:,selectedprotos]
        #print  "s2array"
        s2array = np.squeeze(np.asarray(np.sum(temp, axis=1))) 
        #print s2array.shape
        #print type(s2array)
        
        #print "s2array after ratio"
        s2array = s2array / (len(selectedprotos))
        #print s2array.shape
        #print type(s2array)
        
        #print  "s1array after subtraction"  
        s1array = np.abs(s1array - s2array)
        #print s1array.shape
        #print type(s1array)
        
        
        if reg == 'logdet':
            #print "we are regularising"
            if inverse_of_prev_selected is not None: # first call has been made already
                temp = K_subset[candidates, :][:, candidates2[selected]]
                #print "temp"
                #print type(temp)
                #print temp.shape
                
                K_subset2 = K_subset[candidates,:][:,candidates2[candidates]]
                #print "K_subset2"
                #print type(K_subset2)
                #print K_subset2.shape
                
                temp2 = np.dot(inverse_of_prev_selected, temp.transpose())
                #temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                #print "temp2"
                #print type(temp2)
                #print temp2.shape
                
                #print "i want to compute the first regulariser"
                regularizer = np.dot(temp, temp2)
                #regularizer = temp * temp2.T
                #print "regularizer"
                #print type(regularizer)
                #print regularizer.shape
                
                regcolsum = np.sum(regularizer, axis=0)
                #print "regcolsum"
                #print type(regcolsum)
                #print regcolsum.shape
                
                regularizer = np.squeeze(np.asarray(np.log(np.abs(np.diagonal(K_subset2))))) - np.squeeze(np.asarray(regcolsum))
                #print "regularizer"
                #print type(regularizer)
                #print regularizer.shape
                
                s1array = s1array + regularizer
                #print "s1array"
                #print type(s1array)
                #print s1array.shape
                
            else:
                #print "we are doing the first step"
                K_subset2 = K_subset[candidates_row,:][:,candidates2]
                #print "K_subset2"
                #print type(K_subset2)
                #print K_subset2.shape
                
                #print "we are doing the first step"
                s1array = s1array - np.log(np.abs(np.diagonal(K_subset2)))
                #print "s1array"
                #print type(s1array)
                #print s1array.shape
                #print s1array
                    
        #print "we are outside regularisation"
        argmax = candidates[np.argmax(s1array)]
        #print "argmax"
        #print argmax
        maxx = np.max(s1array)

        selected = np.append(selected, argmax)
        #print "selected"
        #print selected
        if reg == 'logdet':
            KK = K_subset[selected,:][:,candidates2[selected]]
            #print "KK"
            #print KK.shape
            #print KK
            #print type(KK)
            
            inverse_of_prev_selected = np.linalg.inv(KK) # shortcut
            #print "inverse of prev selected"
            #print inverse_of_prev_selected
            #print type(inverse_of_prev_selected)

    return candidates2[selected]
