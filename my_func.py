# tesselation and transition probability matrix
# Urszula Golyska 2022
import numpy as np
from scipy import sparse
import networkx as nx
import graphviz as gv
from modularity_maximization import spectralopt

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
    # extr_id = tess_ind[np.argmax(abs(x[:,ex_dim]-m)),:]

    # extreme event - within >5 sigma away from the mean
    dev = np.std(x[:,ex_dim])
    extr_id = tess_ind[abs(x[:,ex_dim])>=m+5*dev,:] # define extreme event as three times the standard deviation away from the mean

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

    # translate points into lexicographic order
    row = tess_to_lexi(P[:,:dim],N, dim)
    col = tess_to_lexi(P[:, dim:], N, dim)
    extr_trans = tess_to_lexi(np.array(extr_id), N, dim)

    P = sparse.coo_matrix((data, (row, col)), shape=(N**dim, N**dim)) # create sparse matrix

    return P, extr_trans    # return sparse probability matrix with points in lexicographic order and the extreme event point

def community_aff(P_com_old, P_com_new, N, dim, type, printing):
    """Creates a community affiliation matrix D, in which each node or old cluster is matched with the new cluster they
    were assigned to in the previous step

     :param P_com_old: clustered community P
    :param P_com_new: refined and reclustered community P
    :param N: number of discretsations in each direction
    :param dim: dimensions of the system
    :param type: 'first' or 'iteration', defines whether we are clustering clusters or nodes
    :param printing: bool parameter if the communities and their nodes should be printed on screen
    :return: returns a dense Dirac matrix of the affiliation of points to the identified clusters
    """
    nr_com_new = int(np.size(np.unique(np.array(list(P_com_new.values())))))
    if type=='iteration':
        nr_com_old = int(np.size(np.unique(np.array(list(P_com_old.values())))))
        D = np.zeros((nr_com_old, nr_com_new))
    elif type=='first':
        D = np.zeros((N ** dim, nr_com_new))  # number of points by number of communities
    if printing:
        # print all communities and their node entries
        print('Total number of new communities: ', nr_com_new)

    for com in np.unique(np.array(list(P_com_new.values()))):   # for all communities
        if printing:
            print("Community: ", com)
            print("Nodes: ", end='')
        if type=='iteration':
            for key, value in P_com_old.items():    # loop through all communities
                if value == com:
                    if printing:
                        print(key, end=', ')    # print nodes in the community
                    D[value,com] = 1  # to prescribe nodes to communities
        elif type=='first':
            for key, value in P_com_new.items():  # loop through all communities
                if value == com:
                    if printing:
                        print(key, end=', ')  # print nodes in the community
                    D[key, value] = 1  # to prescribe nodes to communities
        if printing:
            print('')
    return D

def community_aff_sparse(P_com_old, P_com_new, N, dim, type, printing):
    """Creates a sparse community affiliation matrix D, in which each node or old cluster is matched with the new cluster they
    were assigned to in the previous step

    :param P_com_old: clustered community P
    :param P_com_new: refined and reclustered community P
    :param N: number of discretsations in each direction
    :param dim: dimensions of the system
    :param type: 'first' or 'iteration', defines whether we are clustering clusters or nodes
    :param printing: bool parameter if the communities and their nodes should be printed on screen
    :return: returns a sparse Dirac matrix of the affiliation of points to the identified clusters
    """
    D = np.empty((0,3), dtype=int)  # matrix of indices of sparse matrix
    nr_com_new = int(np.size(np.unique(np.array(list(P_com_new.values())))))

    for com in np.unique(np.array(list(P_com_new.values()))):   # for all communities
        if printing:
            print("Community: ", com)
            print("Nodes: ", end='')
        for key, value in P_com_new.items():  # loop through all communities
            if value == com:
                if printing:
                    print(key, end=', ')  # print nodes in the community
                row = [key, value, 1]  # to prescribe nodes to communities
                D = np.vstack([D, row])
        if printing:
            print('')

    if type=='iteration':
        nr_com_old = int(np.size(np.unique(np.array(list(P_com_old.values())))))
        D_sparse = sparse.coo_matrix((D[:, 2], (D[:,0], D[:,1])), shape=(nr_com_old, nr_com_new))
    elif type=='first':
        D_sparse = sparse.coo_matrix((D[:,2], (D[:,0],D[:,1])), shape=(N ** dim, nr_com_new))
    if printing:
        # print all communities and their node entries
        print('Total number of new communities: ', nr_com_new)


    return D_sparse

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

def to_graph_sparse(P):
    """Translates a sparse probability matrix into graph form

    :param P: sparse transition matrix (directed with weighted edges)
    :return: returns graph version of matrix P
    """
    columns = P.row   # because of the different definition of the P matrix - for us it's P[to, from], for graph for - P[from,to]
    rows = P.col
    data = P.data
    P_graph = nx.DiGraph()
    for i in range(len(columns)):
        P_graph.add_edge(rows[i], columns[i], weight=data[i])   # to check: should it be columns, rows or rows,columns
    return P_graph

def to_graph_gv(P):
    """Translates a sparse probability matrix into graph form for the gv package used for large communities

    :param P: sparse transition matrix (directed with weighted edges)
    :return: returns graph version of matrix P
    """
    columns = P.row   # because of the different definition of the P matrix - for us it's P[to, from], for graph for - P[from,to]
    rows = P.col
    data = P.data
    P_graph = gv.Digraph('G', filename='cluster.gv')
    for i in range(len(columns)):
        P_graph.edge(str(rows[i]), str(columns[i]), label=str(data[i]))
    return P_graph

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
    if type=='classic': #account for the last point
        corr_point = tess_ind[-1]   #coreection point
    elif type=='backwards': #account for the first point
        corr_point = tess_ind[0]

    for j in range(len(u_1[:, 0])):  # for each unique entry (each tesselation box)
        point_1 = u_1[j]  # index of the point j (in current box)
        denom = counts_1[j]  # denominator for the probability
        if (point_1==corr_point).all(): # check if current point is the one of interest
            denom=denom-1
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

def extr_iden(extr_trans, D_nodes_in_clusters, P_old):
    """Identifies clusters of extreme event and its predecessor

    :param extr_trans: node identified before as extreme event
    :param D_nodes_in_clusters: matrix of affiliation of point to the community clusters
    :param P_old: transition probability matrix
    :return: returns tuple of cluster of the extreme event and the clusters of its predecessor
    """
    ## Identify extreme event it's precursor
    if type(extr_trans)==np.int32:
        extr_cluster = int(D_nodes_in_clusters.col[D_nodes_in_clusters.row == extr_trans])
        from_cluster = P_old.col[P_old.row == extr_cluster]
    else:
        extr_cluster=[]
        for point in extr_trans:    #all nodes in extreme event
            loc_cluster =  int(D_nodes_in_clusters.col[D_nodes_in_clusters.row==point]) # cluster of point
            if loc_cluster not in extr_cluster:
                extr_cluster.append(loc_cluster)

        from_cluster = []
        for cluster in extr_cluster:    # for all extreme event clusters
            from_cluster_loc = P_old.col[P_old.row==cluster]  # from clusters that transition to extreme clusters
            for loc_cluster in from_cluster_loc:
                if loc_cluster not in from_cluster:
                    from_cluster.append(loc_cluster)
    from_cluster = np.delete(from_cluster,np.where(from_cluster == extr_cluster))  # remove iteration within extreme cluster

    # nodes_from = []
    # nodes_to = []
    # if type=='bifurcation':
    #     extreme_from = np.where(np.count_nonzero(P1.transpose(), axis=1) > 2)  # will give the row
    #     extreme_from = int(extreme_from[0])
    #     extreme_to = np.where(P1.transpose()[extreme_from, :] == P1.transpose()[extreme_from, P1.transpose()[extreme_from,
    #                                                                                       :].nonzero()].min())  # indentifies clusters from and to which we have the extreme event transition
    #     extreme_to = int(extreme_to[0])
    # if type=='deviation':
    #     nodes_to = [extr_trans]  # in this approach this is NOT in the clustered form
    #     extreme_to = P_community[extr_trans]  # cluster
    #     extreme_from = P1[extreme_to, :].nonzero()  # cluster from which we can transition to extreme event
    #     extreme_from = extreme_from[0]
    #     if extreme_to in extreme_from:  # remove the option of transitions within the cluster
    #         extreme_from = np.delete(extreme_from, np.where(extreme_from == extreme_to))
    #     # if np.size(extreme_from)==1:
    #         # extreme_from = int(extreme_from)
    #     if np.size(extreme_from)>1:
    #         print("More than one cluster transitioning to extreme event found:", extreme_from)
    #
    # for key, value in P_community.items():
    #     if value in extreme_from:
    #         nodes_from.append(key)
    #     if value == extreme_to:
    #         if (type=='deviation' and key not in nodes_to) or type=='bifurcation':
    #             nodes_to.append(key)

    return (extr_cluster,from_cluster)     #(extreme_from, extreme_to, nodes_from, nodes_to)

def clustering_loop(P_community_old, P_graph_old, P_old, D_nodes_in_clusters):
    """Runs the loop of re-clustering (for a iterative process)

    :param P_community_old: previous community, to be compressed
    :param P_graph_old: previous community in graph form
    :param P_old: previous transition probability matrix
    :param D_nodes_in_clusters: matrix of affiliation of point to the previous community clusters
    :return: returns a set of the same parameters, but after one optimization run, can be then directly fed to the
     function again to obtain more optimized results
    """

    P_community_new = spectralopt.partition(P_graph_old)  # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
    D_new = community_aff(P_community_old, P_community_new, 0, 0, 'iteration', 1)  # matrix of point-to-cluster affiliation

    # Deflate the Markov matrix
    P_new = np.matmul(np.matmul(D_new.transpose(), P_old.transpose()), D_new)  # P1 transposed or not?
    print(np.sum(P_new,axis=0).tolist())  # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    # P_new = P_new.transpose()   # had to add transpose for the classic probability, why? the same for backwards?
    P_graph_old = to_graph(P_new)
    P_community_old = P_community_new
    P_old = P_new

    # make translation of which nodes belong to the new cluster
    D_nodes_in_clusters = np.matmul(D_nodes_in_clusters, D_new)
    return P_community_old, P_graph_old, P_old, D_nodes_in_clusters

def clustering_loop_sparse(P_community_old, P_graph_old, P_old, D_nodes_in_clusters):
    """Runs the loop of re-clustering (for a iterative process); sparse version

    :param P_community_old: previous community, to be compressed
    :param P_graph_old: previous community in graph form (sparse)
    :param P_old: previous transition probability matrix (sparse)
    :param D_nodes_in_clusters: sparse matrix of affiliation of point to the previous community clusters
    :return: returns a set of the same parameters, but after one optimization run, can be then directly fed to the
     function again to obtain more optimized results
    """

    P_community_new = spectralopt.partition(P_graph_old)  # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
    D_new = community_aff_sparse(P_community_old, P_community_new, 0, 0, 'iteration', 1)  # matrix of point-to-cluster affiliation

    # Deflate the Markov matrix
    P_new = sparse.coo_matrix((D_new.transpose() * P_old) * D_new)
    print(np.sum(P_new,axis=0).tolist())  # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    # P_new = P_new.transpose()   # had to add transpose for the classic probability, why? the same for backwards?
    P_graph_old = to_graph_sparse(P_new)
    P_community_old = P_community_new
    P_old = P_new

    # make translation of which nodes belong to the new cluster
    D_nodes_in_clusters = sparse.coo_matrix(D_nodes_in_clusters*D_new)
    return P_community_old, P_graph_old, P_old, D_nodes_in_clusters

def data_to_clusters(tess_ind_trans, D_nodes_in_clusters):
    '''Translates datapoints to cluster number affiliation

    :param tess_ind_trans: datapoint time series already translated to tesselated lexicographic ordering
    :param D_nodes_in_clusters: matrix of affiliation of all the possible points to current community ordering
    :return: returns vector of the time series with their cluster affiliation
    '''
    tess_ind_cluster = np.zeros_like(tess_ind_trans)
    for point in np.unique(tess_ind_trans):     # take all unique points in tesselated space
        cluster_aff = int(D_nodes_in_clusters.col[D_nodes_in_clusters.row==point])  # find affiliated cluster
        tess_ind_cluster[tess_ind_trans==point] = cluster_aff
    return tess_ind_cluster