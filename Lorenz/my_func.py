# tesselation and transition probability matrix
# Urszula Golyska 2022
import numpy as np
from scipy import sparse

def tesselate(x,N):
    """ Tesselate data points x into space defined by N spaces in each direction

    :param x: vector of point coordinates in consequent time steps
    :param N: number of discretsations in each direction
    :return: returns matrix tess_ind which includes the indices of the box taken by the data points in consequent time steps
    """
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
    return tess_ind    # returns indices of occupied spaces, without values


def trans_matrix(tess_ind):
    """Computes transition probability matrix of tesselated data.

    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :return : returns sparse transition probability matrix P, where a row contains the coordinate of the point i to which
    the transition occurs, point j from which the transition occurs and the value of probability of the transition
    """
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

def prob_to_sparse(P,N):
    """"Translates the transition probability matrix of any dimensions into a python sparse 2D matrix

    :param P: probability transition matrix as described in trans_matrix
    :param N: number of discretsations in each direction
    :return: returns python (scipy) sparse coordinate 2D matrix
    """
    dim = int((np.size(P[0, :])-1)/2)  # dimensions

    data = P[:,-1]  # store probability data in separate vector and delete it from the probability matrix
    P = np.delete(P, -1, axis=1)

    # translate points into lexicographic order
    for i in range(1,dim):  # loop through dimensions
        P[:,i]=P[:,i]*N*i  # first point (to)
        P[:,i+dim]=P[:,i+dim]*N*i   # second point (from)

    row = np.sum(P[:,:dim], axis=1)
    col = np.sum(P[:, dim:],axis=1)

    P = sparse.coo_matrix((data, (row, col)), shape=(N**dim, N**dim)) # create sparse matrix
    return P