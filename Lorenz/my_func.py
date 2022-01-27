# tesselation and transition probability matrix
# Urszula Golyska 2022
import numpy as np

def tesselate(x,N):
    dim = int(np.size(x[0,:])) # dimensions
    # dx = 1/N*np.ones(dim)
    y = np.zeros_like(x)
    tess_ind = np.empty((0,dim), dtype=int)  # matrix od indices of sparse matrix

    for i in range(dim):
        y[:,i] = np.divide((x[:,i]-min(x[:,i])),abs(max(x[:,i])-min(x[:,i])))   # rescaling in all dimensions to [0,1]

    # start from lower left corner (all (rescaled) dimensions = 0)
    for k in range(np.size(x[:,0])): # loop through all points
        point_ind = np.floor(y[k,:]*N).astype(int)  # vector of indices of the given point, rounding down
        point_ind[point_ind==N] = N-1   # for all point located at the very end (max) - put them in the last cell
        tess_ind = np.vstack([tess_ind, point_ind])   # sparse approach, translate the points into the indices of the tesselation
        # to get the tesselated space, just take the unique rows of tess_ind
    return  tess_ind    # returns indices of occupied spaces, without values


def trans_matrix(tess_ind):
    dim = int(np.size(tess_ind[0, :]))  # dimensions
    P = np.empty((0,2*dim+1))   # probability matrix dim*2+1 for the value of the probability P[0,:] = [to_index(3), from_index(3), prob_value(1)]
    u_to,index_to, counts_to = np.unique(tess_ind, axis=0, return_index = True, return_counts = True)  # sorted points that are occupied at some point

    for i in range(len(u_to[:,0])):     # for each unique entry (each tesselation box)
        point_to = u_to[i]     # index of the point i (in current box)
        denom = counts_to[i]    # denominator for the probability
        temp = np.all(tess_ind==point_to,axis=1)   # rows of tess_ind with point i
        temp = np.append(temp[1:], [False])   # indices of the row just above (j); adding a false to the end
        u_from, index_from, counts_from = np.unique(tess_ind[temp], axis=0, return_index=True, return_counts=True)  # sorted points occupied just before going to i

        for j in range(len(counts_from)):  # loop through all instances of i
            point_from = u_from[j]
            temp = np.append([[point_to], [point_from]], [counts_from[j] / denom])
            P = np.vstack([P, temp ]) # add row to sparse probability matrix

    # eliminate the initial conditions if its there - is this done?
    return P    # returns sparse transition probability matrix in the form (i,j,k,l,m,n,p[ij])
