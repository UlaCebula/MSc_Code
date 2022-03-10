# tesselation and transition probability matrix
# Urszula Golyska 2022
import numpy as np
from scipy import sparse
import networkx as nx

def tesselate(x,N,ex_dim):
    """ Tesselate data points x into space defined by N spaces in each direction

    :param x: vector of point coordinates in consequent time steps
    :param N: number of discretsations in each direction
    :param ex_dim: dimension by which the extreme event should be identified
    :return: returns matrix tess_ind which includes the indices of the box taken by the data points in consequent time steps,
    and the index of the identified extreme event
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

    # Find tesselated index of extreme event
    m = np.mean(x[:, ex_dim])   # mean of chosen parameter
    extr_id = tess_ind[np.argmax(abs(x[:,ex_dim]-m)),:]

    return tess_ind, extr_id    # returns indices of occupied spaces, without values and the index of the identified extreme event

def tess_to_lexi(x,N,dim):
    """Translated tesselated space of any dimensions to lexicographic order

    :param x: array of tesselated space coordinates
    :param N: number of discretsations in each direction
    :param dim: dimensions of the phase space
    :return: returns 1d array of tesselated space
    """
    x2 = np.zeros_like(x)
    for i in range(dim):  # loop through dimensions
        if x.size>dim:    # if we have more than one point
            x2[:,i]=x[:,i]*N**i
        else:
            x2[i] = x[i]*N**i

    if x.size>dim:    # if we have more than one point
        x_trans = np.sum(x2[:,:dim], axis=1)
    else:
        x_trans = np.sum(x2[:dim])

    return x_trans

def prob_to_sparse(P,N, extr_id):
    """"Translates the transition probability matrix of any dimensions into a python sparse 2D matrix

    :param P: probability transition matrix as described in trans_matrix
    :param N: number of discretsations in each direction
    :param extr_id: index in the tesselated space of the identified extreme event
    :return: returns python (scipy) sparse coordinate 2D matrix and the translated index of the extreme event
    """
    dim = int((np.size(P[0, :])-1)/2)  # dimensions

    data = P[:,-1]  # store probability data in separate vector and delete it from the probability matrix
    P = np.delete(P, -1, axis=1)
    # extr_trans = extr_id

    # translate points into lexicographic order
    row = tess_to_lexi(P[:,:dim],N, dim)
    col = tess_to_lexi(P[:, dim:], N, dim)
    extr_trans = tess_to_lexi(np.array(extr_id), N, dim)
    # for i in range(1,dim):  # loop through dimensions
    #     P[:,i]=P[:,i]*N*i  # first point (to)
    #     P[:,i+dim]=P[:,i+dim]*N*i   # second point (from)
    #     extr_trans[i] = extr_trans[i]*N*i
    #
    # row = np.sum(P[:,:dim], axis=1)
    # col = np.sum(P[:, dim:],axis=1)
    # extr_trans = np.sum(extr_trans)

    P = sparse.coo_matrix((data, (row, col)), shape=(N**dim, N**dim)) # create sparse matrix

    return P, extr_trans    # return sparse probability matrix with points in lexicographic order and the extreme event point

def community_aff(P_com, N, dim, printing):
    """Creates a community affiliation matrix D, in which each point is matched with the cluster they were assigned to in the previous step

    :param P_com: clustered community P
    :param N: number of discretsations in each direction
    :param dim: dimensions of the system
    :param priting: bool parameter if the communities and their nodes should be printed on screen
    :return: returns a dense Dirac matrix of the affiliation of points to the identified clusters
    """
    nr_communities = int(np.size(np.unique(np.array(list(P_com.values())))))
    if printing:
        # print all communities and their node entries
        print('Total number of communities: ', nr_communities)

    D = np.zeros((N ** dim, nr_communities))  # number of points by number of communities
    for com in np.unique(np.array(list(P_com.values()))):   # for all communities
        if printing:
            print("Community: ", com)
            print("Nodes: ", end='')
        for key, value in P_com.items():    # loop through all communities
            if value == com:
                if printing:
                    print(key, end=', ')    # print nodes in the community
                D[key, value] = 1  # to prescribe nodes to communities
        if printing:
            print('')
    return D

def community_aff_clusters(P_com_old, P_com_new, printing):
    """Creates a community affiliation matrix D, in which each old community is matched with the new cluster they were assigned to in the previous step

    :param P_com_old: clustered community P
    :param P_com_new: refined and reclustered community P
    :param printing: bool parameter if the communities and their nodes should be printed on screen
    :return: returns a dense Dirac matrix of the affiliation of old clusters to the newly identified clusters
    """

    nr_com_old = int(np.size(np.unique(np.array(list(P_com_old.values())))))
    nr_com_new = int(np.size(np.unique(np.array(list(P_com_new.values())))))
    if printing:
        # print all communities and their node entries
        print('Total number of new communities: ', nr_com_new)

    D = np.zeros((nr_com_old, nr_com_new))
    for com in np.unique(np.array(list(P_com_new.values()))):   # for all new communities
        if printing:
            print("Community: ", com)   # new community
            print("Nodes: ", end='')    # old communities belonging to said new community
        for key, value in P_com_old.items():    # loop through all old communities
            if value == com:
                if printing:
                    print(key, end=', ')    # print nodes in the community
                D[value, com] = 1  # to prescribe nodes to communities
        if printing:
            print('')
    return D

def to_graph(P):
    """Translates a probability matrix into graph form

    :param P: transition matrix (directed with weighted edges)
    :return: returns graph version of matrix P
    """
    P = P.transpose()   # because of the different definition of the P matrix - for us it's P[to, from], for graph for - P[from,to]
    P_graph = nx.DiGraph()
    for i in range(len(P[:, 0])):
        for j in range(len(P[0, :])):
            if P[i, j] != 0:
                P_graph.add_edge(i, j, weight=P[i, j])
    return P_graph

def prob_classic(tess_ind):
    """Calculates the transition probability matrix of tesselated data in classic sense.

    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :return: returns sparse probability matrix P, where a row contains the coordinate of the point i to which
    the transition occurs, point j from which the transition occurs and the value of probability of the transition
    """
    dim = int(np.size(tess_ind[0, :]))  # dimensions
    P = np.empty((0,2 * dim + 1))  # probability matrix dim*2+1 for the value of the probability ...
    # P[0,:] = [to_index(dim), from_index(dim), prob_value(1)]
    u_from, index_from, counts_from = np.unique(tess_ind, axis=0, return_index=True,
                                          return_counts=True)  # sorted points that are occupied at some point

    for j in range(len(u_from[:, 0])):  # for each unique entry (each tesselation box)
        point_from = u_from[j]  # index of the point j (in current box)
        denom = counts_from[j]  # denominator for the probability
        temp = np.all(tess_ind == point_from, axis=1)  # rows of tess_ind with point j
        temp = np.append([False], temp[:-1])  # indices of the row just below (i); adding a false to the beginning
        u_to, index_to, counts_to = np.unique(tess_ind[temp], axis=0, return_index=True,
                                                    return_counts=True)  # sorted points occupied just before going to i

        for i in range(len(counts_to)):  # loop through all instances of i
            point_to = u_to[i]
            temp = np.append([[point_to], [point_from]], [counts_to[i] / denom])
            P = np.vstack([P, temp])  # add row to sparse probability matrix


    return P # returns sparse transition probability matrix in the form (i,j,k,l,m,n,p[ij]); for consistency this will
    # also be in the form of P[to,from], which will have to be transposed later in the code

def prob_backwards(tess_ind):
    """Computes transition probability matrix of tesselated data in the backwards sense, as presented in Schmid (2018).

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

def probability(tess_ind, type):
    """Computes transition probability matrix of tesselated data in both the classic and the backwards sense (Schmid (2018)).

    :param tess_ind:  matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :param type: "classic" - traditional approach of calculating the probability of transitioning from state j to state i,
    "backwards" - calculating probability of having transitioned from point j when already in point i
    :return: returns sparse transition probability matrix P, where a row contains the coordinate of the point i to which
    the transition occurs, point j from which the transition occurs and the value of probability of the transition
    """
    # for type = 'backwards' 1 is to, 2 is from; for type = 'classic', 1 is from, 2 is to
    dim = int(np.size(tess_ind[0, :]))  # dimensions
    P = np.empty((0, 2 * dim + 1))  # probability matrix dim*2+1 for the value of the probability
                                    # P[0,:] = [to_index(dim), from_index(dim), prob_value(1)]
    u_1, index_1, counts_1 = np.unique(tess_ind, axis=0, return_index=True,
                                                return_counts=True)  # sorted points that are occupied at some point

    for j in range(len(u_1[:, 0])):  # for each unique entry (each tesselation box)
        point_1 = u_1[j]  # index of the point j (in current box)
        denom = counts_1[j]  # denominator for the probability
        temp = np.all(tess_ind == point_1, axis=1)  # rows of tess_ind with point j
        if type=='classic':
            temp = np.append([False], temp[:-1])  # indices of the row just below (i); adding a false to the beginning
        elif type=='backwards':
            temp = np.append(temp[1:], [False])  # indices of the row just above (j); adding a false to the end
        u_2, index_2, counts_2 = np.unique(tess_ind[temp], axis=0, return_index=True,
                                              return_counts=True)  # sorted points occupied just before going to i

        for i in range(len(counts_2)):  # loop through all instances of i
            point_2 = u_2[i]
            if type=='classic':
                temp = np.append([[point_2], [point_1]], [counts_2[i] / denom])
            elif type=='backwards':
                temp = np.append([[point_1], [point_2]], [counts_2[i] / denom])
            P = np.vstack([P, temp])  # add row to sparse probability matrix

    return P


def extr_iden_bif(P1,P_community):
    """Finds the indices of the communities between which the transition to the extreme event occurs by identifying
    bifurcation in the graph path

    :param P1: deflated transition probability matrix
    :param P_community: partition community with keys as nodes and values as clusters
    :return: returns tuple  of index of clusters and nodes when the extreme event occurs
    """
    extreme_from = np.where(np.count_nonzero(P1.transpose(), axis=1) > 2)  # will give the row
    extreme_from = int(extreme_from[0])
    extreme_to = np.where(P1.transpose()[extreme_from, :] == P1.transpose()[extreme_from, P1.transpose()[extreme_from,
                                                                                          :].nonzero()].min())  # indentifies clusters from and to which we have the extreme event transition
    extreme_to = int(extreme_to[0])

    nodes_from = []
    nodes_to = []
    for key, value in P_community.items():
        if value == extreme_from:
            nodes_from.append(key)
        if value == extreme_to:
            nodes_to.append(key)

    return (extreme_from, extreme_to, nodes_from, nodes_to)

def extr_iden_amp(P1,P_community,extr_trans):
    """Finds the indices of the communities between which the transition to the extreme event occurs based on the previously
    identified node with maximum variance

    :param P1: deflated transition probability matrix
    :param P_community: partition community with keys as nodes and values as clusters
    :param extr_trans: the translated index of the extreme event (in lexicographic order)
    :return: returns tuple of index of clusters and nodes when the extreme event occurs
    """
    nodes_from = []
    nodes_to = [extr_trans]   # in this approach this is NOT in the clustered form
    extreme_to = P_community[extr_trans]    # cluster
    # other way around - from_trans will be the point that can translate to extr_trans (from P1 after clustering)
    extreme_from = P1[extreme_to,:].nonzero()   # cluster from which we can transition to extreme event
    extreme_from = extreme_from[0]
    if extreme_to in extreme_from:      # remove the option of transitions within the cluster
        extreme_from=np.delete(extreme_from, np.where(extreme_from==extreme_to))

    for key, value in P_community.items():
        if value == extreme_from:
            nodes_from.append(key)
        if value == extreme_to and key not in nodes_to:
            nodes_to.append(key)

    return (int(extreme_from), extreme_to, nodes_from, nodes_to)

def extr_iden(P1,P_community,type, extr_trans):
    """Identifies nodes and clusters of extreme event and its predecessor using the bifurcation or max deviation method

    :param P1: deflated transition probability matrix
    :param P_community: partition community with keys as nodes and values as clusters
    :param type: defines method of computation: 'bifurcation' or 'deviation'
    :param extr_trans: the translated index of the extreme event (in lexicographic order)
    :return: returns tuple of index of clusters and nodes when the extreme event occurs
    """
    nodes_from = []
    nodes_to = []
    if type=='bifurcation':
        extreme_from = np.where(np.count_nonzero(P1.transpose(), axis=1) > 2)  # will give the row
        extreme_from = int(extreme_from[0])
        extreme_to = np.where(P1.transpose()[extreme_from, :] == P1.transpose()[extreme_from, P1.transpose()[extreme_from,
                                                                                          :].nonzero()].min())  # indentifies clusters from and to which we have the extreme event transition
        extreme_to = int(extreme_to[0])
    if type=='deviation':
        nodes_to = [extr_trans]  # in this approach this is NOT in the clustered form
        extreme_to = P_community[extr_trans]  # cluster
        extreme_from = P1[extreme_to, :].nonzero()  # cluster from which we can transition to extreme event
        extreme_from = extreme_from[0]
        if extreme_to in extreme_from:  # remove the option of transitions within the cluster
            extreme_from = np.delete(extreme_from, np.where(extreme_from == extreme_to))
        extreme_from = int(extreme_from)

    for key, value in P_community.items():
        if value == extreme_from:
            nodes_from.append(key)
        if value == extreme_to:
            if (type=='deviation' and key not in nodes_to) or type=='bifurcation':
                nodes_to.append(key)

    return (extreme_from, extreme_to, nodes_from, nodes_to)