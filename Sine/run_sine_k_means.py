# comparison case  using k-means clustering for a simple sine wave case
# Urszula Golyska 2022

from sklearn.cluster import KMeans
from my_func import *

def sine_data_generation(t0, tf, dt, nt_ex, rand_threshold=0.9, rand_amplitude=2, rand_scalar=1):
    """Function for generating data of the sine wave system

    :param t0: starting time, default t0=0
    :param tf: final time
    :param dt: time step
    :param nt_ex: number of time steps that the extreme event will last
    :param rand_threshold: threshold defining the probability of the extreme event
    :param rand_amplitude: amplitude defining the jump of the extreme event
    :param rand_scalar: scalar value defining the size of the extreme event
    :return: returns time vector t and matrix of data (of size 2*Nt)
    """
    # time discretization
    t = np.arange(t0,tf,dt)

    # phase space
    u1 = np.sin(t)
    u2 = np.cos(t)

    # generate random spurs
    for i in range(len(t)):
        if u1[i]>=0.99 and abs(u2[i])<=0.015:
            if np.random.rand() >=rand_threshold:
                u1[i:i+nt_ex] = rand_scalar*u1[i:i+nt_ex]+rand_amplitude
                u2[i:i+nt_ex] = rand_scalar*u2[i:i+nt_ex]+rand_amplitude

    u = np.hstack([np.reshape(u1, (len(t),1)),np.reshape(u2, (len(t),1))])  # combine into one matrix
    return t, u

plt.close('all') # close all open figures
type='sine'

# time discretization
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex = 50  # number of time steps of the extreme event

# extreme event parameters
rand_threshold = 0.9
rand_amplitude = 2
rand_scalar = 1

t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)

extr_dim = [0,1]   # define both phase space coordinates as extreme event

# Tesselation
M = 20
nr_dev = 2
prob_type='classic'
nr_clusters_vec=[2,5,8,10,15,20,27]

##########LOOP START#################
for nr_clusters in nr_clusters_vec:
    dim = x.shape[1]
    tess_ind, extr_id = tesselate(x, M, extr_dim,nr_dev)  # where extr_dim indicates the dimension by which the extreme event should be identified
    # Transition probability
    P = probability(tess_ind, prob_type)  # create sparse transition probability matrix

    # plot_time_series(x,t,type)
    # plot_phase_space(x,type)
    # plot_tesselated_space(tess_ind, type),

    tess_ind_trans = tess_to_lexi(tess_ind, M, dim)
    P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

    # Graph form
    P_graph = to_graph_sparse(P)  # translate to dict readable for partition

    # Visualize unclustered graph
    # plot_graph(P_graph,False,type)

    # Clustering--
    kmeans = KMeans(init="k-means++",n_clusters=nr_clusters, n_init=10,max_iter=300,random_state=42)
    kmeans.fit(x)

    D = np.empty((0,3), dtype=int)  # matrix of indices of sparse matrix

    nodes, indices = np.unique(tess_ind_trans, return_index=True)

    for i in range(len(nodes)):   # for all unique nodes
        row = [nodes[i], kmeans.labels_[indices[i]], 1]  # to prescribe nodes to communities
        D = np.vstack([D, row])

    D_sparse = sp.coo_matrix((D[:,2], (D[:,0],D[:,1])), shape=(M ** dim, nr_clusters))

    # Deflate the Markov matrix
    P1 = sp.coo_matrix((D_sparse.transpose() * P) * D_sparse)
    print(np.sum(P1,axis=0).tolist())  # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P1_graph = to_graph(P1.toarray())
    # plot_graph(P1_graph,True, type)

    # color palette - linear blue
    palette = plt.get_cmap('viridis', D_sparse.shape[1])

    # translate datapoints to cluster number affiliation
    tess_ind_cluster = kmeans.labels_

    # cluster centers
    coord_clust_centers, coord_clust_centers_tess = cluster_centers(x,tess_ind, tess_ind_cluster, D_sparse,dim)
    coord_clust_centers = kmeans.cluster_centers_

    # identify extreme clusters and those transitioning to them
    extr_clusters, from_clusters = extr_iden(extr_trans, D_sparse, P1)

    print('From cluster: ', from_clusters, 'To extreme cluster: ', extr_clusters)

    for i in range(P1.shape[0]): # for all unique rows of the deflated probability matrix
        denom = np.sum(D_sparse,axis=0)
        denom = denom[0,i]  # sum of nodes in cluster - we should divide by this
        P1.data[P1.row == i] = P1.data[P1.row == i]/denom

    # plot_time_series_clustered(x[:,0], t, tess_ind_cluster, palette, type)

    # Visualize phase space trajectory with clusters
    plot_phase_space_clustered(x, type, D_sparse, tess_ind_cluster, coord_clust_centers, extr_clusters,nr_dev, palette)

    #plot tesselated phase space with clusters
    plot_phase_space_tess_clustered(tess_ind, type, D_sparse, tess_ind_cluster, coord_clust_centers_tess, extr_clusters, palette)

    # Visualize probability matrix
    # plot_prob_matrix(P1.toarray())

    # list of class type objects
    clusters = []

    # define individual properties of clusters:
    for i in range(D_sparse.shape[1]):   # loop through all clusters
        nodes = D_sparse.row[D_sparse.col==i]

        center_coord=coord_clust_centers[i,:]
        center_coord_tess=coord_clust_centers_tess[i,:]

        # average time spend in cluster
        avg_time, nr_instances = avg_time_in_cluster(i,tess_ind_cluster,t)

        clusters.append(cluster(i, nodes, center_coord, center_coord_tess, avg_time, nr_instances, P1, extr_clusters, from_clusters))

plt.show()

#########LOOP END###############




# plotting = True
# min_clusters=15
# max_it=5
# clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic', 2,plotting, True)
# calculate_statistics(extr_dim, clusters, P, tf)
# plt.show()