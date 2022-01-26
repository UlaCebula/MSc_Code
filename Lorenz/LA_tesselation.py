# tesselation
# Urszula Golyska 2022
import numpy as np

def tesselate(x,N):
    dim = int(np.size(x[0,:])) #dimensions
    dx = 1/N*np.ones(dim)
    y = np.zeros_like(x)
    tess_ind = np.empty((0,dim), dtype=int)  # matrix od indices of sparse matrix
    # A_mtrx = np.zeros(N*np.ones(dim, dtype=int)) #non-sparse approach

    for i in range(dim):
        y[:,i] = np.divide((x[:,i]-min(x[:,i])),abs(max(x[:,i])-min(x[:,i])))   # rescaling

    # start from lower left corner (all (rescaled) dimensions = 0)
    for k in range(np.size(x[:,0])): # loop through all points
        point_ind = np.floor(y[k,:]*N).astype(int)  # vector of indices of the given point
        point_ind[point_ind==N] = -1   # for all point located at the very end (max) - put them in the last cell
        if point_ind not in tess_ind:
            tess_ind=np.vstack([tess_ind, point_ind])   # sparse approach
        # A_mtrx[point_ind[0], point_ind[1], point_ind[2]]=1    # non-sparse approach, for some reason A_mtrx[point_ind] does not work
    return  tess_ind #A_mtrx returns indices of occupied spaces, without values

