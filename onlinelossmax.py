from __future__ import division  #So that when we divide by integers we gegt a float. Eg so that 5/2 = 2.5 not 2.
import numpy as np


def online_loss_max(stored_phis, full_phi_sum, proto_phi_sum, prototype_indices, criticism_indices, n, M):
    #ONLINE_LOSS_MAX maximizes L(C) online.
    #Here we are adding x_n and assume that stored_phis and full_phi_sum have been updated to include x_new: the new data point.

    criticism_indices = criticism_indices + [n-1]
    phis_for_criticisms = stored_phis[criticism_indices]
    
    losses = np.zeros(M+1)
    for l in range(M+1):
        losses[l] = np.abs((1/n)*np.inner(full_phi_sum, phis_for_criticisms[l]) - (1/M)*np.inner(proto_phi_sum, phis_for_criticisms[l]))
    
    min_loss_idx = np.argmin(losses)
    min_loss = losses[min_loss_idx]
    new_loss = losses[M]

    if new_loss > min_loss:
        criticism_indices = np.delete(criticism_indices, min_loss_idx)
    else:
        criticism_indices = np.delete(criticism_indices, M+1)
    
    return criticism_indices

# total_loss = sum(losses)


#def online_greedy_loss_max(stored_phis, full_phi_sum, proto_phi_sum, prototype_indices, criticism_indices, n, M, greedy = False):
#    C = []
#    while len(C) < M:
#        for i in (range(n)-prototype_indices