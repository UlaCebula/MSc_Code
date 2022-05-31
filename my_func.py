# tesselation and transition probability matrix
# Urszula Golyska 2022

import numpy as np
import scipy.sparse as sp
import networkx as nx
import graphviz as gv
from modularity_maximization import spectralopt
import matplotlib.pyplot as plt

class cluster(object):
    def __init__(self, nr, nodes, center_coord,center_coord_tess,avg_time, P, extr_clusters, from_clusters):
        self.nr = nr    # number of cluster
        self.nodes = nodes  #hypercubes - id in tesselated space

        is_extreme = False
        is_precursor = False
        if nr in extr_clusters:
            is_extreme=True
        if nr in from_clusters:
            is_precursor=True

        self.is_extreme = is_extreme
        self.is_precursor = is_precursor

        clusters_to = P.row[P.col==nr]   # clusters to which we can go from this one
        clusters_from = P.col[P.row==nr]     # clusters from which we can get to this one

        self.transition_to = clusters_to
        self.transition_from = clusters_from

        # definition of the boundaries - weird shape
        self.center = center_coord
        self.center_tess = center_coord_tess

        # average time send in cluster
        self.avg_time = avg_time

        max_prob = 0
        for extr_cluster in extr_clusters:
            if nr in P.col[P.row==extr_cluster]: #reverse probability matrix - if our cluster transitions directly to extr_cluster
                temp = P.data[P.col[P.row==extr_cluster]]
                loc_prob = temp[P.col[P.row==extr_cluster]==nr]
                # loc_prob = loc_prob/ # DIVIDE BY SUM OF ROW!!
                if loc_prob>max_prob:
                    max_prob = loc_prob

        self.prob_to_extreme = max_prob

def tesselate(x,N,ex_dim,nr_dev=7):
    """ Tesselate data points x into space defined by N spaces in each direction

    :param x: vector of point coordinates in consequent time steps
    :param N: number of discretsations in each direction
    :param ex_dim: dimensions by which the extreme event should be identified
    :param nr_dev: scalar defining how far away from the mean (multiples of the standard deviation) will be considered an extreme event
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
    m = np.zeros_like(ex_dim,dtype=float)
    dev = np.zeros_like(ex_dim,dtype=float)
    # Find tesselated index of extreme event
    for i in range(len(ex_dim)):
        loc_ex_dim = ex_dim[i]
        m[i] = np.mean(x[:, loc_ex_dim])   # mean of chosen parameter
        dev[i] = np.std(x[:,loc_ex_dim])

        # extreme event - within >=nr_dev sigma away from the mean
        if i==0:    # first dimension
            temp = abs(x[:,ex_dim[i]])>=m[i]+nr_dev*dev[i] # define extreme event as nr_dev the standard deviation away from the mean
        else:   # for other dimensions - delete from the vector that we already obtained
            temp = np.logical_and((temp==True), abs(x[:, ex_dim[i]]) >= m[i] + nr_dev * dev[i])
    if len(ex_dim)>0:   # if we have an extreme event (we are supposed to have)
        extr_id = tess_ind[temp,:]
    else:
        extr_id=[]

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
    if len(extr_id)!=0:
        extr_trans = tess_to_lexi(np.array(extr_id), N, dim)
    else:
        extr_trans=0

    P = sp.coo_matrix((data, (row, col)), shape=(N**dim, N**dim)) # create sparse matrix

    return P, extr_trans    # return sparse probability matrix with points in lexicographic order and the extreme event point

def find_least_probable(P,n,M):
    P_dense = P.toarray()
    P_dense = P_dense.flatten()
    least_prob_tess = np.zeros((n,4))

    least_prob = np.argsort(P_dense)[:n]
    # for i in range(len(least_prob)):
    #     flat_id = least_prob[i] # index of transformation in flattened prob
    #     P_id =  # index of transformation in normal prob
    #     least_prob_tess[i] =    # index of transformation in tesselated space
    shape = (400,400)
    least_prob = np.unravel_index(least_prob, shape)
    least_prob = np.stack((least_prob[0],least_prob[1]))

    for i in range(n):  # for all points
        # translate least probable states to tesselated space
        least_prob_tess[i,:] = [int(least_prob[0,i]/M),least_prob[0,i]%M,int(least_prob[1,i]/M),least_prob[1,i]%M]
    return least_prob_tess

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
        D_sparse = sp.coo_matrix((D[:, 2], (D[:,0], D[:,1])), shape=(nr_com_old, nr_com_new))
    elif type=='first':
        D_sparse = sp.coo_matrix((D[:,2], (D[:,0],D[:,1])), shape=(N ** dim, nr_com_new))
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

def plot_graph(P_graph):
    """Function for plotting the graph representation of the probability matrix

    :param P_graph: graph form of the probability matrix
    :return: none, plots graph representation of probability matrix
    """
    # Visualize graph
    plt.figure()
    nx.draw_kamada_kawai(P_graph,with_labels=True)
    return 1

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

def plot_prob_matrix(P_dense):
    """Function for plotting probability matrix

    :param P_dense: dense representation of calculated probability matrix
    :return: none, plots probability matrix
    """
    # Visualize probability matrix
    plt.figure(figsize=(7, 7))
    plt.imshow(P_dense,interpolation='none', cmap='binary')
    plt.colorbar()
    return 1

def extr_iden(extr_trans, D_nodes_in_clusters, P_old):
    """Identifies clusters of extreme event and its predecessor

    :param extr_trans: node identified before as extreme event
    :param D_nodes_in_clusters: matrix of affiliation of point to the community clusters
    :param P_old: transition probability matrix
    :return: returns tuple of cluster of the extreme event and the clusters of its predecessor
    """
    ## Identify extreme event it's precursor
    if type(extr_trans)==np.int32 or type(extr_trans)==int:
        extr_cluster = D_nodes_in_clusters.col[D_nodes_in_clusters.row == extr_trans]
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
    # from_cluster = np.delete(from_cluster,np.where(from_cluster in extr_cluster))  # remove iteration within extreme cluster - this doesn't work correctly, but I will rewrite this function anyways


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
    P_new = sp.coo_matrix((D_new.transpose() * P_old) * D_new)
    print(np.sum(P_new,axis=0).tolist())  # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    # P_new = P_new.transpose()   # had to add transpose for the classic probability, why? the same for backwards?
    P_graph_old = to_graph_sparse(P_new)
    P_community_old = P_community_new
    P_old = P_new

    # make translation of which nodes belong to the new cluster
    D_nodes_in_clusters = sp.coo_matrix(D_nodes_in_clusters*D_new)
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

def cluster_centers(x,tess_ind, tess_ind_cluster, D_nodes_in_clusters,dim):
    coord_clust = np.zeros((D_nodes_in_clusters.shape[1], dim))
    for i in range(D_nodes_in_clusters.shape[1]):   # for each cluster
        coord_clust[i,:] = np.mean(x[tess_ind_cluster==i,:], axis=0)

    pts, indices = np.unique(tess_ind, return_index=True, axis=0)      #unique points
    coord_clust_tess = np.zeros((D_nodes_in_clusters.shape[1], dim))
    num_clust = np.zeros((D_nodes_in_clusters.shape[1], 1))

    for i in range(len(pts[:, 0])):  # for each unique point
        loc_clust = tess_ind_cluster[indices[i]]
        num_clust[loc_clust] += 1
        coord_clust_tess[loc_clust,:] += pts[i, :]

    for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
        coord_clust_tess[i,:] = coord_clust_tess[i,:] / num_clust[i]

    return coord_clust, coord_clust_tess

def plot_phase_space(x, type):
    """Function for plotting the MFE data in phase space

    :param x: data matrix (look at to_burst or read_DI)
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots data x in equivalent phase space
    """
    if type=='MFE_burst':

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='MFE_dissipation':
        plt.figure()
        plt.plot(x[:,1], x[:,0])
        plt.title("Dissipation vs energy")
        plt.ylabel("D")
        plt.xlabel("I")

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.plot(x[:,0],x[:,1])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("$x$")
        plt.ylabel("$y$")

    if type=='LA':
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz Attractor")

    if type=='CDV':
        plt.figure(figsize=(6, 6))
        plt.scatter(x[:, 0], x[:, 3])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        plt.figure(figsize=(6,6))
        plt.plot(x[:,2], x[:,4])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    return 1

def plot_tesselated_space(tess_ind,type, least_prob_tess=[0]):
    """Function for plotting the MFE data in tesselated phase space

    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots data x in equivalent tesselated phase space
    """
    if type=='MFE_burst':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:, 0], tess_ind[:, 1], tess_ind[:, 2])
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")

    if type=='MFE_dissipation':
        plt.figure(figsize=(7, 7))
        plt.scatter(tess_ind[:,1], tess_ind[:,0], s=200, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
        if np.size(least_prob_tess)>1:
            for i in range(len(least_prob_tess[:,0])):  # for all least probable events
                plt.plot([least_prob_tess[i,0],least_prob_tess[i,2]], [least_prob_tess[i,1],least_prob_tess[i,3]], '--r')
                plt.scatter(least_prob_tess[i,0],least_prob_tess[i,1], facecolors = 'red', edgecolor = 'red')     # from
                plt.scatter(least_prob_tess[i,2], least_prob_tess[i,3], facecolors='green', edgecolor='green')    # to (or maybe the other way around)
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("I")
        plt.ylabel("D")

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.scatter(tess_ind[:, 0], tess_ind[:, 1], s=200, marker='s', facecolors='None',
                    edgecolor='blue')  # I should relate somehow s to N and the fig size
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("x")
        plt.ylabel("y")

    if type=='LA':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:,0], tess_ind[:,1], tess_ind[:,2])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if type=='CDV':
        plt.figure(figsize=(6, 6))
        plt.scatter(tess_ind[:, 0], tess_ind[:, 3], s=200, marker='s', facecolors='None',
                    edgecolor='blue')
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        plt.figure(figsize=(6, 6))
        plt.scatter(tess_ind[:, 2], tess_ind[:, 4], s=200, marker='s', facecolors='None',
                    edgecolor='blue')
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    return 1

def plot_phase_space_clustered(x,type,D_nodes_in_clusters,tess_ind_cluster, coord_centers, extr_cluster,nr_dev,palette):
    """Function for plotting phase space with cluster affiliation

    :param x: data matrix (look at to_burst or read_DI)
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :param D_nodes_in_clusters: matrix of affiliation of point to the community clusters
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :return: none, plots the phase space colored by cluster affiliation
    """
    if type=='MFE_burst':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2])  # I should relate somehow s to N and the fig size
            if i in extr_cluster:
                ax.text(coord_centers[i,0], coord_centers[i,1], coord_centers[i,2], str(i),color='r')  # numbers of clusters
            else:
                ax.text(coord_centers[i,0], coord_centers[i,1], coord_centers[i,2], str(i))  # numbers of clusters
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='MFE_dissipation':
        plt.figure(figsize=(7, 7))
        plt.axhline(y=np.mean(x[:,0])+nr_dev*np.std(x[:, 0]), color='r', linestyle='--') # plot horizontal cutoff
        plt.axvline(x=np.mean(x[:, 1]) + nr_dev*np.std(x[:, 1]), color='r', linestyle='--')  # plot horizontal cutoff
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i,1],
                        x[tess_ind_cluster == i,0], c=palette[i,:])  # I should relate somehow s to N and the fig size

            if i in extr_cluster:     # if cluster is extreme - plot number in red
                plt.text(coord_centers[i,1], coord_centers[i,0], str(i),color='r')  # numbers of clusters
            else:
                plt.text(coord_centers[i,1], coord_centers[i,0], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("I")
        plt.ylabel("D")

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.axhline(y=np.mean(x[:,0])+nr_dev*np.std(x[:, 0]), color='r', linestyle='--') # plot horizontal cutoff
        plt.axvline(x=np.mean(x[:,1]) + nr_dev*np.std(x[:, 1]), color='r', linestyle='--')  # plot horizontal cutoff
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i,0],
                        x[tess_ind_cluster == i,1], c=palette[i,:])  # I should relate somehow s to N and the fig size
            if i in extr_cluster:      # if cluster is extreme - plot number in red
                plt.text(coord_centers[i,0], coord_centers[i,1], str(i),color='r')  # numbers of clusters
            else:
                plt.text(coord_centers[i,0], coord_centers[i,1], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("x")
        plt.ylabel("y")

    if type=='LA':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter3D(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2],c=palette[i,:])  # I should relate somehow s to N and the fig size
            if i in extr_cluster:
                ax.text(coord_centers[i,0], coord_centers[i,1],coord_centers[i,2], str(i), color='r')  # numbers of clusters
            else:
                ax.text(coord_centers[i,0], coord_centers[i,1],coord_centers[i,2], str(i))  # numbers of clusters
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if type=='CDV':
        plt.figure(figsize=(6, 6))
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i, 0],
                        x[tess_ind_cluster == i, 3], c=palette[i, :])  # I should relate somehow s to N and the fig size
            if i in extr_cluster:  # if cluster is extreme - plot number in red
                plt.text(coord_centers[i, 0], coord_centers[i, 3], str(i), color='r')  # numbers of clusters
            else:
                plt.text(coord_centers[i, 0], coord_centers[i, 3], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        plt.figure(figsize=(6,6))
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i, 2],
                        x[tess_ind_cluster == i, 4], c=palette[i, :])  # I should relate somehow s to N and the fig size
            if i in extr_cluster:  # if cluster is extreme - plot number in red
                plt.text(coord_centers[i, 2], coord_centers[i, 4], str(i), color='r')  # numbers of clusters
            else:
                plt.text(coord_centers[i, 2], coord_centers[i, 4], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    return 1

def plot_phase_space_tess_clustered(tess_ind, type, D_nodes_in_clusters, tess_ind_cluster, coord_centers_tess, extr_cluster, palette):
    if type=='MFE_dissipation':
        # take only unique spots in tesselated space
        x,indices=np.unique(tess_ind,return_index=True,axis=0)

        plt.figure(figsize=(7, 7))
        for i in range(len(x[:,0])): #for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette[loc_clust,:]
            plt.scatter(x[i,1], x[i,0], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col) #I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_cluster:
                plt.text(coord_centers_tess[i,1], coord_centers_tess[i,0], str(i),color='r')  # numbers of clusters
            else:
                plt.text(coord_centers_tess[i,1], coord_centers_tess[i,0], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("I")
        plt.ylabel("D")

    if type=='sine':
        # take only unique spots in tesselated space
        x,indices=np.unique(tess_ind,return_index=True,axis=0)

        plt.figure(figsize=(7, 7))
        for i in range(len(x[:,0])): #for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette[loc_clust,:]
            plt.scatter(x[i,0], x[i,1], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col) #I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_cluster:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,1], str(i),color='r')  # numbers of clusters
            else:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,1], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("x")
        plt.ylabel("y")

    if type=='CDV':
        # take only unique spots in tesselated space
        x, indices = np.unique(tess_ind, return_index=True, axis=0)

        plt.figure(figsize=(6, 6))
        for i in range(len(x[:,0])): #for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette[loc_clust,:]
            plt.scatter(x[i,0], x[i,3], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col) #I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_cluster:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,3], str(i),color='r')  # numbers of clusters
            else:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,3], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        # take only unique spots in tesselated space
        x, indices = np.unique(tess_ind, return_index=True, axis=0)

        plt.figure(figsize=(6, 6))
        for i in range(len(x[:, 0])):  # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette[loc_clust, :]
            plt.scatter(x[i, 2], x[i, 4], s=200, marker='s', facecolors=loc_col,
                        edgecolor=loc_col)  # I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_cluster:
                plt.text(coord_centers_tess[i, 2], coord_centers_tess[i, 4], str(i), color='r')  # numbers of clusters
            else:
                plt.text(coord_centers_tess[i, 2], coord_centers_tess[i, 4], str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    return 1

def plot_time_series(x,t, type):
    """Function for plotting the time series of MFE data

    :param x: data matrix (look at to_burst or read_DI)
    :param t: time vector
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots time series of data (without plt.show())
    """
    if type=='MFE_burst':
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the MFE flow")
        plt.subplot(3,1,1)
        plt.plot(t,x[:,0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("Roll & streak")
        plt.xlabel("t")
        plt.subplot(3,1,2)
        plt.plot(t,x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("Mean shear")
        plt.xlabel("t")
        plt.subplot(3,1,3)
        plt.plot(t,x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("Burst")
        plt.xlabel("t")

    if type=='MFE_dissipation':
        fig, axs = plt.subplots(2)
        fig.suptitle("Dynamic behavior of the MFE flow")
        plt.subplot(2, 1, 1)
        plt.plot(t, x[:, 0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("D")
        plt.xlabel("t")
        plt.subplot(2, 1, 2)
        plt.plot(t, x[:, 1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("I")
        plt.xlabel("t")

    if type=='sine':
        fig, axs = plt.subplots(2)
        fig.suptitle("Dynamic behavior of the sine wave")
        plt.subplot(2,1,1)
        plt.plot(t, x[:,0])
        plt.ylabel("$x$")
        plt.xlabel("t")
        plt.subplot(2,1,2)
        plt.plot(t, x[:,1])
        plt.ylabel("$y$")
        plt.xlabel("t")

    if type=='LA':
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the Lorenz Attractor")
        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.ylabel("$x$")
        plt.xlabel("t")
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.ylabel("$y$")
        plt.xlabel("t")
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.ylabel("$z$")
        plt.xlabel("t")

    if type=='CDV':

        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the CDV flow")
        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.xlim([t[0],t[-1]])
        plt.ylabel("$x_1$")
        plt.xlabel("t")
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_2$")
        plt.xlabel("t")
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_3$")
        plt.xlabel("t")

        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the CDV flow")

        plt.subplot(3, 1, 1)
        plt.plot(t, x[:, 3])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_4$")
        plt.xlabel("t")
        plt.subplot(3, 1, 2)
        plt.plot(t, x[:, 4])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_5$")
        plt.xlabel("t")
        plt.subplot(3, 1, 3)
        plt.plot(t, x[:, 5])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_6$")
        plt.xlabel("t")

    if type=='PM':
        # Visualize dataset
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the PM flow")

        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_1$")
        plt.xlabel("t")
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_2$")
        plt.xlabel("t")
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_3$")
        plt.xlabel("t")

        fig, axs = plt.subplots(2)
        fig.suptitle("Dynamic behavior of the PM flow")

        plt.subplot(2,1,1)
        plt.plot(t, x[:,3])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_4$")
        plt.xlabel("t")
        plt.subplot(2,1,2)
        plt.plot(t, x[:,4])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_5$")
        plt.xlabel("t")

    return 1

def plot_time_series_clustered(y,t, tess_ind_cluster, palette, type):
    """Function for plotting the time series of data with cluster affiliation

    :param y: vector of the parameter to plot; for type=="burst" y=burst; for type=="dissipation" y=D
    :param t: time vector
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param palette: color palette decoding a unique color code for each cluster
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots time series with cluster affiliation
    """
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:   # if change of cluster
            loc_col = palette[tess_ind_cluster[i]]
            # plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
            plt.scatter((t[i] + t[i + 1]) / 2, (y[i] + y[i + 1]) / 2,marker='s', facecolors = 'None', edgecolor = loc_col)
            # plt.text(t[i], y[i], str(tess_ind_cluster[i]))  # numbers of clusters
    if type=='MFE_burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='MFE_dissipation':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    if type=='sine':
        plt.title("$x$ vs time")
        plt.ylabel("$x$")
    if type == 'LA':
        plt.title("$x$ vs time")
        plt.ylabel("$x$")
    if type=='CDV':
        plt.title("$x_0$ vs time")
        plt.ylabel("$x_0$")
    if type=='PM':
        plt.title("$x_5$ vs time")
        plt.ylabel("$x_5$")
    plt.xlabel("t")

    return 1

def plot_time_series_extr_iden(y,t, tess_ind_cluster, from_cluster, extr_cluster, type):
    '''Function for plotting time series with extreme event and precursor identification

    :param y: vector of the parameter to plot; for type=="burst" y=burst; for type=="dissipation" y=D
    :param t: time vector
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param from_cluster: vector of cluster numbers which can transition to extreme event
    :param extr_cluster: vector of cluster numbers which contain extreme events
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots time series with extreme event (blue) and precursor (red) identification
    '''
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i] in from_cluster and tess_ind_cluster[i+1] in extr_cluster:
            plt.scatter(t[i], y[i], marker='s', facecolors = 'None', edgecolor = 'blue')
            plt.scatter(t[i+1], y[i+1], marker='s', facecolors='None', edgecolor='red')
    if type=='MFE_burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='MFE_dissipation':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    plt.xlabel("t")
    return 1

def avg_time_in_cluster(cluster_id,tess_ind_cluster,t):
    # find all instances of cluster
    ind = np.where(tess_ind_cluster==cluster_id) #returns index
    ind=ind[0]

    nr_cycles = 1
    t_cluster=[]

    t_start= t[ind[0]]
    for i in range(len(ind)-1):
        if ind[i+1]!=ind[i]+1:  # if the next time is not also there
            t_cluster.append(t[ind[i]]-t_start)     # time spend in cluster during cycle
            nr_cycles+=1
            t_start = t[ind[i+1]]
    # include last point
    t_cluster.append(t[ind[-1]] - t_start)
    avg_time = np.mean(t_cluster)

    return avg_time

def extreme_event_identification_process(t,x,dim,M,extr_dim,type, min_clusters, max_it, prob_type='classic',nr_dev=7,plotting=True, first_refined=False):
    """Big loop with calculation for the MFE system enclosed

    :param t: time vector
    :param x: data matrix (look at to_burst or read_DI)
    :param dim: integer, number of dimensions of the system (2 for type=="dissipation" and 3 for type=="burst")
    :param M: number of tesselation discretisations per dimension
    :param extr_dim: dimension which amplitude will define the extreme event (0 for type=="dissipation" and 2 for type=="burst")
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :param type: string defining the type of probability to be calculated, either "classic" (default) or "backwards"
    :param nr_dev: scalar defining how far away from the mean (multiples of the standard deviation) will be considered an extreme event
    :return: none, runs calculations and plots results; can be modified to output the final deflated probability matrix
    """
    tess_ind, extr_id = tesselate(x, M, extr_dim,nr_dev)  # where extr_dim indicates the dimension by which the extreme event should be identified
    # Transition probability
    P = probability(tess_ind, prob_type)  # create sparse transition probability matrix

    if plotting:
        plot_time_series(x,t,type)
        plot_phase_space(x,type)
        plot_tesselated_space(tess_ind, type)

    tess_ind_trans = tess_to_lexi(tess_ind, M, dim)
    P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

    # Graph form
    P_graph = to_graph_sparse(P)  # translate to dict readable for partition

    # Find 5 least probable transitions
    # least_prob_tess = find_least_probable(P,5,M)

    if plotting:
        # Visualize unclustered graph
        plot_graph(P_graph)
        if dim<4:  # the matrix will be too big
            # Visualize probability matrix
            plot_prob_matrix(P.toarray())

    # Clustering
    P_community = spectralopt.partition(P_graph, refine=first_refined)  # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
    D_sparse = community_aff_sparse(0, P_community, M, dim, 'first', 1)  # matrix of point-to-cluster affiliation

    # Deflate the Markov matrix
    P1 = sp.coo_matrix((D_sparse.transpose() * P) * D_sparse)
    print(np.sum(P1,axis=0).tolist())  # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P1_graph = to_graph(P1.toarray())

    # more iterations
    P_community_old = P_community
    P_old = P1
    P_graph_old = P1_graph
    D_nodes_in_clusters = D_sparse
    int_id = 0

    # Deflation and refinement loop
    while int(np.size(np.unique(np.array(list(P_community_old.values()))))) > min_clusters and int_id < max_it:  # condition
        int_id = int_id + 1
        P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old,
                                                                                          P_old, D_nodes_in_clusters)
        print(np.sum(D_nodes_in_clusters,axis=0).tolist())

    if plotting:
        # Visualize clustered graph
        plot_graph(P_graph_old)

        # color palette
        palette = np.zeros((D_nodes_in_clusters.shape[1],3))
        for i in range(D_nodes_in_clusters.shape[1]):
            palette[i,:] = np.random.rand(1,3)

    # translate datapoints to cluster number affiliation
    tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

    # cluster centers
    coord_clust_centers, coord_clust_centers_tess = cluster_centers(x,tess_ind, tess_ind_cluster, D_nodes_in_clusters,dim)

    # identify extreme clusters and those transitioning to them
    extr_clusters, from_clusters = extr_iden(extr_trans, D_nodes_in_clusters, P_old)
    print('From cluster: ', from_clusters, 'To extreme cluster: ', extr_clusters)

    # DIVIDE P BY SUM OF ROW!!
    for i in range(P_old.shape[0]): # for all unique rows of the deflated probability matrix
        denom = np.sum(D_nodes_in_clusters,axis=0)
        denom = denom[0,i]  # sum of nodes in cluster - we should divide by this
        P_old.data[P_old.row == i] = P_old.data[P_old.row == i]/denom

    if plotting:
        # Plot time series with clusters
        if type=='MFE_burst':
            plot_time_series_clustered(x[:,2], t, tess_ind_cluster, palette, type)
        if type=='MFE_dissipation' or type=='sine'or type=='LA' or type=='CDV':
            plot_time_series_clustered(x[:,0], t, tess_ind_cluster, palette, type)
        if type == 'PM':
            plot_time_series_clustered(x[:, 4], t, tess_ind_cluster, palette, type)

        # Visualize phase space trajectory with clusters
        plot_phase_space_clustered(x, type, D_nodes_in_clusters, tess_ind_cluster, coord_clust_centers, extr_clusters,nr_dev, palette)

        # Plot time series with extreme event identification
        # if type == 'burst':
        #     plot_time_series_extr_iden(x[:,2], t, tess_ind_cluster, from_cluster, extr_cluster, type)
        # if type == 'dissipation':
        #     plot_time_series_extr_iden(x[:,0], t, tess_ind_cluster, from_cluster, extr_cluster, type)

        #plot tesselated phase space with clusters
        plot_phase_space_tess_clustered(tess_ind, type, D_nodes_in_clusters, tess_ind_cluster, coord_clust_centers_tess, extr_clusters, palette)

        # Visualize probability matrix
        plot_prob_matrix(P_old.toarray())

    # list of class type objects
    clusters = []

    # define individual properties of clusters:
    for i in range(D_nodes_in_clusters.shape[1]):   # loop through all clusters
        nodes = D_nodes_in_clusters.row[D_nodes_in_clusters.col==i]

        center_coord=coord_clust_centers[i,:]
        center_coord_tess=coord_clust_centers_tess[i,:]

        # average time spend in cluster
        avg_time = avg_time_in_cluster(i,tess_ind_cluster,t)

        clusters.append(cluster(i, nodes, center_coord, center_coord_tess, avg_time, P_old, extr_clusters, from_clusters))

    # for obj in clusters:
    #     print(obj.nr, obj.nodes, obj.center, obj.is_extreme)

    return clusters, D_nodes_in_clusters, P_old      # returns list of clusters (class:cluster), matrix of cluster affiliation and deflated probability matrix (sparse)