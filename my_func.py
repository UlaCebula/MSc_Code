# tesselation and transition probability matrix
# Urszula Golyska 2022

import numpy as np
import scipy.sparse as sp
import networkx as nx
import graphviz as gv
from modularity_maximization import spectralopt
import matplotlib.pyplot as plt
import numpy.linalg
import csv

def find_extr_paths_loop(P,local_path, cluster_from, ready_paths, extr_clusters):
    ''' Inner function for finding loops leading to given extreme cluster

    :param P: sparse deflated probability matrix
    :param local_path: local path leading to extreme cluster (backwards, first is the extreme cluster)
    :param cluster_from: cluster from which we will look deeper into the probability matrix
    :param ready_paths: ready path, where we have made a full circle
    :param extr_clusters: vector of extreme clusters
    :return: vector of ready paths (tuples)
    '''
    next_clusters = P.col[P.row == cluster_from]
    next_clusters = np.delete(next_clusters, np.where(next_clusters==cluster_from))  # exclude looping inside oneself

    for next_cluster in next_clusters:  # look at all paths
        if next_cluster not in extr_clusters:   # if the next one is not extreme
            loc_local_path = local_path # reset
            if next_cluster not in loc_local_path:
                loc_local_path.append(next_cluster)  # append and go deeper
                find_extr_paths_loop(P, loc_local_path, next_cluster,ready_paths, extr_clusters)
            else:
                if tuple(loc_local_path) not in ready_paths:    # if we don't have this path yet - add to vector of all paths
                    ready_paths.append(tuple(loc_local_path))
        else:       # if the next one is extreme
            if tuple(local_path) not in ready_paths:  # if we don't have this path yet - add to vector of all paths
                ready_paths.append(tuple(local_path))
    return ready_paths

def find_extr_paths(extr_clusters,P):
    '''Outer function for finding loops leading to given extreme cluster

    :param extr_clusters: vector of extreme clusters
    :param P: sparse deflated probability matrix
    :return: returns vector of ready paths (tuples) for all extreme clusters
    '''

    final_paths =list()
    for extr_cluster in extr_clusters:

        clusters_from = P.col[P.row==extr_cluster]
        clusters_from = np.delete(clusters_from,np.where(clusters_from==extr_cluster))   # exclude looping inside oneself

        for cluster_from in clusters_from:  # first one
            if cluster_from not in extr_clusters:
                local_path = [extr_cluster, cluster_from]  # start/restart path
                ready_paths = []

                find_extr_paths_loop(P, local_path, cluster_from, ready_paths, extr_clusters)       # we don't have to add the if statement before this one because it's the first try and we excluded it already
                final_paths.extend(ready_paths)
    return final_paths

def prob_to_extreme(cluster_nr,paths, T, P, clusters):
    '''Function to find the maximum probability and minimum average time to an extreme event

    :param cluster_nr: number of cluster we are currently looking at
    :param paths: vector of ready paths (tuples) for all extreme clusters
    :param T: maximum time of data series
    :param P: sparse deflated probability matrix
    :param clusters: all defined clusters with their properties
    :return: return maximum probability, minimum average time and shortest path to an extreme event for the given cluster_nr cluster
    '''
    prob = 0
    time = T
    length = np.size(P)
    if clusters[cluster_nr].is_extreme ==2: # extreme cluster
        prob = 1
        time = 0
        length = 0
    else:
        for i in range(len(paths)):     # for all paths
            loc_prob = 1
            loc_time = 0
            loc_path = np.asarray(paths[i])
            if cluster_nr in loc_path:     # find path with our cluster
                #take into account only part of path to our cluster
                loc_end = np.where(loc_path==cluster_nr)[0]
                loc_end = loc_end[0]
                loc_path = loc_path[0:loc_end+1]

                for j in range(len(loc_path)):
                    if j!=len(loc_path)-1:  # skip last step, because we don't want to add that
                        temp = P.data[P.col[P.row == loc_path[j]]]     # row is to, col is from
                        temp = temp[P.col[P.row == loc_path[j]]==loc_path[j+1]]
                        loc_prob = loc_prob*temp

                    if j!=0:   # exclude first and last path
                        loc_time += clusters[loc_path[j]].avg_time

                if loc_prob > prob:
                    prob = loc_prob
                if loc_time < time:
                    time = loc_time
                if len(loc_path)-1<length:
                    length = len(loc_path)-1
    return prob,time,length

class cluster(object):
    '''Object cluster, defined by it's number, the nodes that belong to it, it's center, the clusters to and from which
        it transitions'''
    def __init__(self, nr, nodes, center_coord,center_coord_tess,avg_time, nr_instances, P, extr_clusters, from_clusters):
        self.nr = nr    # number of cluster
        self.nodes = nodes  #hypercubes - id in tesselated space

        is_extreme = 0  # not extreme event or precursor
        if nr in extr_clusters:
            is_extreme=2    # extreme event
        elif nr in from_clusters:
            is_extreme=1  # precursor

        self.is_extreme = is_extreme

        clusters_to = P.row[P.col==nr]   # clusters to which we can go from this one
        clusters_from = P.col[P.row==nr]     # clusters from which we can get to this one

        self.transition_to = clusters_to
        self.transition_from = clusters_from

        # definition of the boundaries - weird shape
        self.center = center_coord
        self.center_tess = center_coord_tess

        # average time send in cluster
        self.avg_time = avg_time

        # number of times the cluster loop in given time series
        self.nr_instances = nr_instances

def plot_cluster_statistics(clusters, T, min_prob=None, min_time=None, length=None):
    ''' Function for plotting cluster statistics, for systems than have or do not have extreme events

    :param clusters: all defined clusters with their properties
    :param T: last time of data series
    :param min_prob: maximum probability of transitioning to an extreme event, default is None (for systems without extreme events)
    :param min_time: minimum average time of transitioning to an extreme event, default is None (for systems without extreme events)
    :param length: shortest path to an extreme event, default is None (for systems without extreme events)
    :return: none, makes and saves statistics plots
    '''
    numbers = np.arange(len(clusters))
    color_pal = ['#1f77b4'] * (max(cluster.nr for cluster in clusters)+1)   # default blue

    if min_prob is not None:# if we have extreme events - omit extreme clusters in some of the statistics
        # set color palette
        is_extreme = np.array([cluster.is_extreme for cluster in clusters])
        for i in range(len(is_extreme)):    # not efficient but other ways don't work and idk why
            if is_extreme[i]==2:
                color_pal[i] = '#d62728'    # red
            elif is_extreme[i]==1:
                color_pal[i] = '#ff7f0e'    # orange

        # show cluster statistics
        min_prob[min_prob==1] = 0# change the probability in extreme clusters to zero so they don't appear in the plots
        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(numbers, min_prob, color=color_pal)
        ax.grid('minor')
        temp_labels = ax.containers[0]
        for p in temp_labels:
            x, y = p.get_xy()
            w, h = p.get_width(), p.get_height()
            if h != 0:  # anything that have a height of 0 will not be annotated
                ax.text(x + 0.5 * w, y + h, '%0.2e' % h, va='bottom', ha='center')
        # ax.bar_label(ax.containers[0], label_type='edge')
        ax.set_xlabel("Cluster", fontsize=20)
        ax.set_ylabel("Probability of transitioning to extreme", fontsize=20)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
        plt.savefig('stat_prob.pdf')

        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(numbers, np.round(min_time,2), color=color_pal)
        ax.grid('minor')
        temp_labels = ax.containers[0]
        for p in temp_labels:  # skip the last patch as it is the background
                x, y = p.get_xy()
                w, h = p.get_width(), p.get_height()
                if h != 0:  # anything that have a height of 0 will not be annotated
                    ax.text(x + 0.5 * w, y + h, '%0.2f' % h, va='bottom', ha='center')
        # ax.bar_label(temp_labels, label_type='edge')
        ax.set_xlabel("Cluster", fontsize=20)
        ax.set_ylabel("Average time to extreme [s]", fontsize=20)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
        plt.savefig('stat_time.pdf')

        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(numbers, length, color=color_pal)
        ax.grid('minor')
        temp_labels = ax.containers[0]
        for p in temp_labels:  # skip the last patch as it is the background
            x, y = p.get_xy()
            w, h = p.get_width(), p.get_height()
            if h != 0:  # anything that have a height of 0 will not be annotated
                ax.text(x + 0.5 * w, y + h, '%0.0f' % h, va='bottom', ha='center')
        # ax.bar_label(ax.containers[0], label_type='edge')
        ax.set_xlabel("Cluster", fontsize=20)
        ax.set_ylabel("Length of shortest path to extreme", fontsize=20)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
        plt.savefig('stat_path.pdf')

    # average time in cluster
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [np.round(cluster.avg_time,2) for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("Average time spend in cluster [s]", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_circ_time.pdf')

    # percentage of total time spend in cluster
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [np.round((cluster.avg_time*cluster.nr_instances)/T*100,3) for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("% time spend in cluster", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_percent_time.pdf')

    # cluster size (in nodes)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [cluster.nodes.size for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("# nodes in cluster", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_nr_nodes.pdf')

    # number of instances in whole data frame
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(numbers, [cluster.nr_instances for cluster in clusters], color=color_pal)
    ax.grid('minor')
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_xlabel("Cluster", fontsize=20)
    ax.set_ylabel("# instances in time series", fontsize=20)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99)
    plt.savefig('stat_nr_instances.pdf')

    # write statistics to csv file
    with open('cluster_stat.csv', 'w') as file:
        writer = csv.writer(file)
        if min_prob is not None:    # we have extreme events
            csv_header = ['nr', 'avg_time_in_cluster', 'percent_time_in_cluster', 'nr_nodes', 'nr_instances', 'is_extreme', 'prob_to_extreme', 'avg_time_to_extreme', 'path_to_extreme']
            writer.writerow(csv_header)
            for i in numbers:
                csv_data = [i, clusters[i].avg_time, (clusters[i].avg_time*clusters[i].nr_instances)/T*100, clusters[i].nodes.size, clusters[i].nr_instances, is_extreme[i], min_prob[i], min_time[i], length[i]]
                writer.writerow(csv_data)
        else:
            csv_header = ['nr', 'avg_time_in_cluster', 'percent_time_in_cluster', 'nr_nodes', 'nr_instances']
            writer.writerow(csv_header)
            for i in numbers:
                csv_data = [i, clusters[i].avg_time, (clusters[i].avg_time * clusters[i].nr_instances) / T * 100, clusters[i].nodes.size, clusters[i].nr_instances]
                writer.writerow(csv_data)
    return 1

def backwards_avg_time_to_extreme(is_extreme,dt):
    ''' Finds average time of transitioning to precursor cluster that then goes to extreme cluster, looking backwards from extreme cluster

    :param is_extreme: vector defining which clusters are extreme (value 2) and which are precursors (value 1)
    :param dt: time step
    :param clusters: all defined clusters with their properties
    :return: returns value of the average time from entering a precursor stage to the occurrence of an extreme event, number or instances of extreme events with a precursor, number of instances of extreme events and number of instances of precursors
    '''
    # ADD Maybe also calculates number of false positives etc
    extreme_events_t = np.where(is_extreme==2)[0]
    precursors_t = np.where(is_extreme==1)[0]
    instances_extreme_with_precursor=0
    time = 0
    instances_extreme_no_precursor=0
    instances_precursor_no_extreme=0
    instances_precursor_after_extreme=0

    for i in range(len(extreme_events_t)-1):    # look at all extreme events
        # we are looking only at the first one step of each instance
        # isolate first case
        if i==0 or (extreme_events_t[i+1]==extreme_events_t[i]+1 and extreme_events_t[i-1]!=extreme_events_t[i]-1):
            temp_ee_t = extreme_events_t[i] #time step of first extreme step
            if is_extreme[temp_ee_t-1]!=1:
                instances_extreme_no_precursor+=1
            #look at precursors
            for j in range(len(precursors_t)):
                if precursors_t[j]+1 == temp_ee_t: # find the instance we are talking about
                    k=j-1   # start going backwards
                    while k>=0:
                        if precursors_t[k-1] != precursors_t[k]-1 or k==0:  # we have found the end
                            temp_prec_t = precursors_t[k]
                            instances_extreme_with_precursor += 1
                            time += (temp_ee_t - temp_prec_t) * dt
                            break
                        k-=1
                    break

    for i in range(len(precursors_t)-1):    # look at all extreme events
        # we are looking only at the last step of each instance
        # isolate last case
        if i==len(precursors_t)-1 or (precursors_t[i-1]==precursors_t[i]-1 and precursors_t[i+1]!=precursors_t[i]+1):
            temp_prec_t = precursors_t[i] #last step of precursor step
            if is_extreme[temp_prec_t+1]!=2:
                instances_precursor_no_extreme+=1

        # look at first instance to see if there is an extreme event before
        if i==len(precursors_t)-1 or (precursors_t[i+1]==precursors_t[i]+1 and precursors_t[i-1]!=precursors_t[i]-1):
            temp_prec_t_first = precursors_t[i] #first step of precursor step
            if is_extreme[temp_prec_t_first-1]==2:
                instances_precursor_after_extreme+=1

    avg_to_extreme = time/instances_extreme_with_precursor

    return avg_to_extreme, instances_extreme_with_precursor, instances_extreme_no_precursor, instances_precursor_no_extreme, instances_precursor_after_extreme

def calculate_statistics(extr_dim, clusters, P, T):
    ''' Function for calculating the extreme event statistics and plotting all of the statistics

    :param extr_dim: dimension which amplitude will define the extreme event
    :param clusters: all defined clusters with their properties
    :param P: sparse deflated probability matrix
    :param T: maximum time of data series
    :return: none, creates new variables used for plotting and statistics calculations
    '''
    if np.size(extr_dim)>0:    # if we have an extreme dimension
        extr_clusters = np.empty(0, int)
        for i in range(len(clusters)):  # print average times spend in extreme clusters
            loc_cluster = clusters[i]
            if loc_cluster.is_extreme == 2:
                extr_clusters = np.append(extr_clusters, i)

        paths = find_extr_paths(extr_clusters, P)

        min_prob = np.zeros((len(clusters)))
        min_time = np.zeros((len(clusters)))
        length = np.zeros((len(clusters)))

        for i in range(len(clusters)):  # for each cluster
            # prob to extreme
            loc_prob, loc_time, loc_length = prob_to_extreme(i, paths, T, P, clusters)
            min_prob[i] = loc_prob
            min_time[i] = loc_time
            length[i] = loc_length

        plot_cluster_statistics(clusters, T, min_prob, min_time, length)
    else:
        plot_cluster_statistics(clusters, T)
    return 1

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

def plot_graph(P_graph, labels, type):
    """Function for plotting the graph representation of the probability matrix

    :param P_graph: graph form of the probability matrix
    :param labels: bool property defining whether the labels should  be displayed
    :return: none, plots graph representation of probability matrix
    """
    # Visualize graph
    plt.figure()
    if type=='sine':
        nx.draw_kamada_kawai(P_graph, with_labels=labels)
    else:
        # nx.draw(P_graph,with_labels=True)
        nx.draw_spring(P_graph,with_labels=labels)

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
        corr_point = tess_ind[-1]   #correction point
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

def data_to_clusters(tess_ind_trans, D_nodes_in_clusters, x=[], *clusters):
    '''Translates datapoints to cluster number affiliation

    :param tess_ind_trans: datapoint time series already translated to tesselated lexicographic ordering
    :param D_nodes_in_clusters: matrix of affiliation of all the possible points to current community ordering
    :param x:
    :param clusters:
    :return: returns vector of the time series with their cluster affiliation
    '''
    tess_ind_cluster = np.zeros_like(tess_ind_trans)
    for point in np.unique(tess_ind_trans):     # take all unique points in tesselated space
        if D_nodes_in_clusters.col[D_nodes_in_clusters.row==point].size>0:  # if we already have this point
            cluster_aff = int(D_nodes_in_clusters.col[D_nodes_in_clusters.row==point])  # find affiliated cluster
        else:   # this point has not been encountered yet
            #find cluster with closest center (physical, not in tesselated space)
            y = x[np.where(tess_ind_trans==point),:]    # the point in physical space
            cluster_aff = find_closest_center(y[0,0,:],clusters[0])    # take one (first of these) point
        tess_ind_cluster[tess_ind_trans==point] = cluster_aff
    return tess_ind_cluster

def cluster_centers(x,tess_ind, tess_ind_cluster, D_nodes_in_clusters,dim):
    ''' Function for calculating the centers of found clusters in both physical and tesselated phase space

    :param x: data matrix (look at to_burst or read_DI)
    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param D_nodes_in_clusters: matrix of affiliation of all the possible points to current community ordering
    :param dim: number of dimensions
    :return: returns two vectors: cluster centers in phase space and in tesselated phase space
    '''
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
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
    :return: none, plots data x in equivalent phase space
    """
    if type=='MFE_burst':

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='kolmogorov':
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        plt.figure()
        plt.plot(x[:,1], x[:,0])
        # plt.title("Dissipation vs energy")
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("D", fontsize=20)
        plt.xlabel("k", fontsize=20)

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.plot(x[:,0],x[:,1])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("$x$", fontsize = 20)
        plt.ylabel("$y$", fontsize = 20)

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
        plt.ylabel("$x_4$", fontsize=20)
        plt.xlabel("$x_1$", fontsize=20)

    if type=='PM':
        plt.figure(figsize=(6,6))
        plt.plot(x[:,2], x[:,4])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$", fontsize=20)
        plt.xlabel("$x_3$", fontsize=20)

    return 1

def plot_tesselated_space(tess_ind,type, least_prob_tess=[0]):
    """Function for plotting the MFE data in tesselated phase space

    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
    :param least_prob_tess: vector of least probable events (optional)
    :return: none, plots data x in equivalent tesselated phase space
    """
    if type=='MFE_burst':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:, 0], tess_ind[:, 1], tess_ind[:, 2])
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")

    if type=='kolmogorov':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:, 0], tess_ind[:, 1], tess_ind[:, 2])
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        plt.figure(figsize=(7, 7))
        plt.scatter(tess_ind[:,1], tess_ind[:,0], s=200, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
        if np.size(least_prob_tess)>1:
            for i in range(len(least_prob_tess[:,0])):  # for all least probable events
                plt.plot([least_prob_tess[i,0],least_prob_tess[i,2]], [least_prob_tess[i,1],least_prob_tess[i,3]], '--r')
                plt.scatter(least_prob_tess[i,0],least_prob_tess[i,1], facecolors = 'red', edgecolor = 'red')     # from
                plt.scatter(least_prob_tess[i,2], least_prob_tess[i,3], facecolors='green', edgecolor='green')    # to (or maybe the other way around)
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("k")
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

def plot_phase_space_clustered(x,type,D_nodes_in_clusters,tess_ind_cluster, coord_centers, extr_clusters,nr_dev,palette):
    """Function for plotting phase space with cluster affiliation

    :param x: data matrix
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
    :param D_nodes_in_clusters: matrix of affiliation of point to the community clusters
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param coord_centers: cluster centers in phase space
    :param extr_clusters: vector of cluster numbers which contain extreme events
    :param nr_dev: scalar defining how far away from the mean (multiples of the standard deviation) will be considered an extreme event
    :param palette: color palette decoding a unique color code for each cluster
    :return: none, plots the phase space colored by cluster affiliation
    """
    if type=='MFE_burst':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2])  # I should relate somehow s to N and the fig size
            if i in extr_clusters:
                t = ax.text(coord_centers[i,0], coord_centers[i,1], coord_centers[i,2], str(i),color='r', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = ax.text(coord_centers[i,0], coord_centers[i,1], coord_centers[i,2], str(i), color='white', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type == 'kolmogorov':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            # if i==4:
            #     ax.scatter(x[tess_ind_cluster == i, 0], x[tess_ind_cluster == i, 1], x[tess_ind_cluster == i, 2],
            #                color='red')  # I should relate somehow s to N and the fig size
            # else:
            ax.scatter(x[tess_ind_cluster == i, 0], x[tess_ind_cluster == i, 1], x[tess_ind_cluster == i, 2],
                           color='#1f77b4')  # I should relate somehow s to N and the fig size
            # ax.scatter(x[tess_ind_cluster == i, 0], x[tess_ind_cluster == i, 1], x[tess_ind_cluster == i, 2],color=palette.colors[i,:])  # I should relate somehow s to N and the fig size
            # if i in extr_clusters:
            #     t = ax.text(coord_centers[i, 0], coord_centers[i, 1], coord_centers[i, 2], str(i),
            #             color='r', backgroundcolor='1')  # numbers of clusters
            #     t.set_bbox(dict(facecolor='black', alpha=0.35))
            # else:
            #     t = ax.text(coord_centers[i, 0], coord_centers[i, 1], coord_centers[i, 2],
            #             str(i), color='white', backgroundcolor='1')  # numbers of clusters
            #     t.set_bbox(dict(facecolor='black', alpha=0.35))
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        plt.figure(figsize=(7, 7))
        plt.axhline(y=np.mean(x[:,0])+nr_dev*np.std(x[:, 0]), color='r', linestyle='--') # plot horizontal cutoff
        plt.axvline(x=np.mean(x[:, 1]) + nr_dev*np.std(x[:, 1]), color='r', linestyle='--')  # plot horizontal cutoff
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            # if i==4:
            #     plt.scatter(x[tess_ind_cluster == i,1], x[tess_ind_cluster == i,0], color='red')  # I should relate somehow s to N and the fig size
            #
            # else:
            plt.scatter(x[tess_ind_cluster == i,1], x[tess_ind_cluster == i,0], color=palette.colors[i,:])  # I should relate somehow s to N and the fig size

            # if i in extr_clusters:     # if cluster is extreme - plot number in red
            #     t = plt.text(coord_centers[i,1], coord_centers[i,0], str(i),color='r', backgroundcolor='1')  # numbers of clusters
            #     t.set_bbox(dict(facecolor='black', alpha=0.35))
            # else:
            #     t = plt.text(coord_centers[i,1], coord_centers[i,0], str(i),color='white', backgroundcolor='1')  # numbers of clusters
            #     t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.minorticks_on()
        plt.xlabel("k")
        plt.ylabel("D")

    if type=='sine':
        plt.figure(figsize=(7, 7))
        plt.axhline(y=np.mean(x[:,0])+nr_dev*np.std(x[:, 0]), color='r', linestyle='--') # plot horizontal cutoff
        plt.axvline(x=np.mean(x[:,1]) + nr_dev*np.std(x[:, 1]), color='r', linestyle='--')  # plot horizontal cutoff
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i,0],
                        x[tess_ind_cluster == i,1], color=palette.colors[i,:])  # I should relate somehow s to N and the fig size
            if i in extr_clusters:      # if cluster is extreme - plot number in red
                t = plt.text(coord_centers[i,0], coord_centers[i,1], str(i),color='r', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers[i,0], coord_centers[i,1], str(i), color='white',backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("x")
        plt.ylabel("y")

    if type=='LA':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter3D(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2],color=palette.colors[i,:])  # I should relate somehow s to N and the fig size
            if i in extr_clusters:
                t = ax.text(coord_centers[i,0], coord_centers[i,1],coord_centers[i,2], str(i), color='r', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = ax.text(coord_centers[i,0], coord_centers[i,1],coord_centers[i,2], str(i),color='white', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if type=='CDV':
        plt.figure(figsize=(6, 6))
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i, 0],
                        x[tess_ind_cluster == i, 3], color=palette.colors[i, :])  # I should relate somehow s to N and the fig size
            if i in extr_clusters:  # if cluster is extreme - plot number in red
                plt.text(coord_centers[i, 0], coord_centers[i, 3], str(i), color='r')  # numbers of clusters
            else:
                plt.text(coord_centers[i, 0], coord_centers[i, 3], str(i), color='white')  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_4$")
        plt.xlabel("$x_1$")

    if type=='PM':
        plt.figure(figsize=(6,6))
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i, 2],
                        x[tess_ind_cluster == i, 4], color=palette.colors[i, :])  # I should relate somehow s to N and the fig size
            if i in extr_clusters:  # if cluster is extreme - plot number in red
                t = plt.text(coord_centers[i, 2], coord_centers[i, 4], str(i), color='r', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers[i, 2], coord_centers[i, 4], str(i), color='white', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

    return 1

def plot_phase_space_tess_clustered(tess_ind, type, D_nodes_in_clusters, tess_ind_cluster, coord_centers_tess, extr_clusters, palette):
    """Function for plotting tesselated phase space with cluster affiliation

    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; 'kolmogorov')
    :param D_nodes_in_clusters: matrix of affiliation of point to the community clusters
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param coord_centers_tess: cluster centers in tesselated phase space
    :param extr_clusters: vector of cluster numbers which contain extreme events
    :param palette: color palette decoding a unique color code for each cluster
    :return: none, plots the tesselated phase space colored by cluster affiliation
    """
    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        # take only unique spots in tesselated space
        x,indices=np.unique(tess_ind,return_index=True,axis=0)

        plt.figure(figsize=(7, 7))
        for i in range(len(x[:,0])): #for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust,:]
            # if i==4:
            #     loc_col='red'
            plt.scatter(x[i,1], x[i,0], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col) #I should relate somehow s to N and the fig size

        # for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
        #     if i in extr_clusters:
        #         t = plt.text(coord_centers_tess[i,1], coord_centers_tess[i,0], str(i),color='r', backgroundcolor='1')  # numbers of clusters
        #         t.set_bbox(dict(facecolor='black', alpha=0.35))
        #     else:
        #         t = plt.text(coord_centers_tess[i,1], coord_centers_tess[i,0], str(i), color='white', backgroundcolor='1')  # numbers of clusters
        #         t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("k")
        plt.ylabel("D")

    if type=='kolmogorov':
        # take only unique spots in tesselated space
        x, indices = np.unique(tess_ind, return_index=True, axis=0)

        plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        for i in range(len(x[:, 0])):  # for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust, :]
            # if loc_clust==4:
            #     loc_col='red'
            # else:
            #     loc_col = '#1f77b4'
            ax.scatter3D(x[i, 0], x[i, 1], x[i,2],color=loc_col)  # I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                t = ax.text(coord_centers_tess[i, 0], coord_centers_tess[i, 1], coord_centers_tess[i, 2], str(i),
                         color='r', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = ax.text(coord_centers_tess[i, 0], coord_centers_tess[i, 1], coord_centers_tess[i, 2], str(i), color='white', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        ax.set_xlabel("D")
        ax.set_ylabel("k")
        ax.set_zlabel("|a(1,4)|")

    if type=='sine':
        # take only unique spots in tesselated space
        x,indices=np.unique(tess_ind,return_index=True,axis=0)

        plt.figure(figsize=(7, 7))
        for i in range(len(x[:,0])): #for each unique point
            loc_clust = tess_ind_cluster[indices[i]]
            loc_col = palette.colors[loc_clust,:]
            plt.scatter(x[i,0], x[i,1], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col) #I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                t = plt.text(coord_centers_tess[i,0], coord_centers_tess[i,1], str(i),color='r', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers_tess[i,0], coord_centers_tess[i,1], str(i), color='white', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
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
            loc_col = palette.colors[loc_clust,:]
            plt.scatter(x[i,0], x[i,3], s=200, marker='s', facecolors = loc_col, edgecolor = loc_col) #I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,3], str(i),color='r')  # numbers of clusters
            else:
                plt.text(coord_centers_tess[i,0], coord_centers_tess[i,3], str(i), color='white')  # numbers of clusters
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
            loc_col = palette.colors[loc_clust, :]
            plt.scatter(x[i, 2], x[i, 4], s=200, marker='s', facecolors=loc_col,
                        edgecolor=loc_col)  # I should relate somehow s to N and the fig size

        for i in range(D_nodes_in_clusters.shape[1]):  # for each cluster
            if i in extr_clusters:
                t = plt.text(coord_centers_tess[i, 2], coord_centers_tess[i, 4], str(i), color='r', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
            else:
                t = plt.text(coord_centers_tess[i, 2], coord_centers_tess[i, 4], str(i), color='white', backgroundcolor='1')  # numbers of clusters
                t.set_bbox(dict(facecolor='black', alpha=0.35))
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.ylabel("$x_5$")
        plt.xlabel("$x_3$")

        if type == 'LA':
            plt.figure()
            ax = plt.axes(projection='3d')
            for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
                ax.scatter3D(x[tess_ind_cluster == i, 0], x[tess_ind_cluster == i, 1], x[tess_ind_cluster == i, 2],
                             color=palette.colors[i, :])  # I should relate somehow s to N and the fig size
                if i in extr_clusters:
                    t = ax.text(coord_centers[i, 0], coord_centers[i, 1], coord_centers[i, 2], str(i), color='r',
                                backgroundcolor='1')  # numbers of clusters
                    t.set_bbox(dict(facecolor='black', alpha=0.35))
                else:
                    t = ax.text(coord_centers[i, 0], coord_centers[i, 1], coord_centers[i, 2], str(i), color='white',
                                backgroundcolor='1')  # numbers of clusters
                    t.set_bbox(dict(facecolor='black', alpha=0.35))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

    return 1

def plot_time_series(x,t, type):
    """Function for plotting the time series of MFE data

    :param x: data matrix
    :param t: time vector
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
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

    if type == 'kolmogorov':
        fig, axs = plt.subplots(3)
        plt.subplot(3, 1, 1)
        plt.plot(t, x[:, 0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("D")
        plt.xlabel("t")
        plt.subplot(3, 1, 2)
        plt.plot(t, x[:, 1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("k")
        plt.xlabel("t")
        plt.subplot(3, 1, 3)
        plt.plot(t, x[:, 2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("|a(1,4)|")
        plt.xlabel("t")

    if type=='MFE_dissipation' or type=='kolmogorov_kD':
        fig, axs = plt.subplots(2)
        if type=='MFE_dissipation':
            fig.suptitle("Dynamic behavior of the MFE flow")
        if type=='kolmogorov':
            fig.suptitle("Dynamic behavior of the Kolmogorov flow")
        plt.subplot(2, 1, 1)
        plt.plot(t, x[:, 0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("D", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(2, 1, 2)
        plt.plot(t, x[:, 1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("k", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='sine':
        fig, axs = plt.subplots(2)
        # fig.suptitle("Dynamic behavior of the sine wave")
        plt.subplot(2,1,1)
        plt.plot(t, x[:,0])
        plt.ylabel("$x$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(2,1,2)
        plt.plot(t, x[:,1])
        plt.ylabel("$y$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='LA':
        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the Lorenz Attractor")
        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.ylabel("$x$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.ylabel("$y$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.ylabel("$z$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='CDV':

        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the CDV flow")
        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.xlim([t[0],t[-1]])
        plt.ylabel("$x_1$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_2$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_3$", fontsize=20)
        plt.xlabel("t", fontsize=20)

        fig, axs = plt.subplots(3)
        fig.suptitle("Dynamic behavior of the CDV flow")

        plt.subplot(3, 1, 1)
        plt.plot(t, x[:, 3])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_4$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3, 1, 2)
        plt.plot(t, x[:, 4])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_5$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3, 1, 3)
        plt.plot(t, x[:, 5])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_6$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    if type=='PM':
        # Visualize dataset
        fig, axs = plt.subplots(3)
        # fig.suptitle("Dynamic behavior of the PM flow")

        plt.subplot(3,1,1)
        plt.plot(t, x[:,0])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_1$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,2)
        plt.plot(t, x[:,1])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_2$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(3,1,3)
        plt.plot(t, x[:,2])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_3$", fontsize=20)
        plt.xlabel("t", fontsize=20)

        fig, axs = plt.subplots(2)
        fig.suptitle("Dynamic behavior of the PM flow")

        plt.subplot(2,1,1)
        plt.plot(t, x[:,3])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_4$", fontsize=20)
        plt.xlabel("t", fontsize=20)
        plt.subplot(2,1,2)
        plt.plot(t, x[:,4])
        plt.xlim([t[0], t[-1]])
        plt.ylabel("$x_5$", fontsize=20)
        plt.xlabel("t", fontsize=20)

    return 1

def plot_time_series_clustered(y,t, tess_ind_cluster, palette, type):
    """Function for plotting the time series of data with cluster affiliation

    :param y: vector of the parameter to plot; for type=="burst" y=burst; for type=="dissipation" y=D
    :param t: time vector
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param palette: color palette decoding a unique color code for each cluster
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
    :return: none, plots time series with cluster affiliation
    """
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:   # if change of cluster
            loc_col = palette.colors[tess_ind_cluster[i]]
            # plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
            plt.scatter((t[i] + t[i + 1]) / 2, (y[i] + y[i + 1]) / 2,marker='s', facecolors = 'None', edgecolor = loc_col)
            # plt.text(t[i], y[i], str(tess_ind_cluster[i]))  # numbers of clusters
    if type=='MFE_burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='MFE_dissipation' or type=='kolmogorov':
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

def plot_time_series_extr_iden(y,t, tess_ind_cluster, from_cluster, extr_clusters, type):
    '''Function for plotting time series with extreme event and precursor identification

    :param y: vector of the parameter to plot; for type=="burst" y=burst; for type=="dissipation" y=D
    :param t: time vector
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param from_cluster: vector of cluster numbers which can transition to extreme event
    :param extr_clusters: vector of cluster numbers which contain extreme events
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
    :return: none, plots time series with extreme event (blue) and precursor (red) identification
    '''
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i] in from_cluster and tess_ind_cluster[i+1] in extr_clusters:
            plt.scatter(t[i], y[i], marker='s', facecolors = 'None', edgecolor = 'blue')
            plt.scatter(t[i+1], y[i+1], marker='s', facecolors='None', edgecolor='red')
    if type=='MFE_burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='MFE_dissipation' or type=='kolmogorov':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    plt.xlabel("t")
    return 1

def avg_time_in_cluster(cluster_id,tess_ind_cluster,t):
    ''' Function for calculating the average time spend in cluster

    :param cluster_id: number of local cluster
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param t: time vector
    :return: returns average time spend in given cluster
    '''
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

    return avg_time, nr_cycles

def find_closest_center(y,clusters):
    min_dist = numpy.linalg.norm(y)
    closest_cluster =-1
    for cluster in clusters:
        dist = numpy.linalg.norm(cluster.center-y)
        if dist<min_dist:
            closest_cluster=cluster.nr
    return closest_cluster

def extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, prob_type='classic',nr_dev=7,plotting=True, first_refined=False):
    """Big loop with calculation for the different systems

    :param t: time vector
    :param x: data matrix (look at to_burst or read_DI)
    :param dim: integer, number of dimensions of the system (2 for type=="dissipation" and 3 for type=="burst")
    :param M: number of tesselation discretisations per dimension
    :param extr_dim: dimensions which amplitude will define the extreme event (0 for type=="dissipation" and 2 for type=="burst")
    :param type: string defining the type of system ("MFE_burst"; "MFE_dissipation"; "sine"; "LA"; "CDV"; "PM"; "kolmogorov")
    :param min_clusters: minimum number of clusters which breaks the deflation loop
    :param max_it: maximum number of iterations which breaks the deflation loop
    :param prob_type: string defining the type of probability to be calculated, either "classic" (default) or "backwards"
    :param nr_dev: scalar defining how far away from the mean (multiples of the standard deviation) will be considered an extreme event
    :param plotting: bool property defining whether to plot the data
    :param first_refined: bool property defining whether the first clustering should be done with refinement
    :return: none, runs calculations and plots results; can be modified to output the final deflated probability matrix
    """
    dim = x.shape[1]
    tess_ind, extr_id = tesselate(x, M, extr_dim,nr_dev)  # where extr_dim indicates the dimension by which the extreme event should be identified
    # Transition probability
    P = probability(tess_ind, prob_type)  # create sparse transition probability matrix

    if plotting:
        plot_time_series(x,t,type)
        plot_phase_space(x,type)
        plot_tesselated_space(tess_ind, type),


    tess_ind_trans = tess_to_lexi(tess_ind, M, dim)
    P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

    # Graph form
    P_graph = to_graph_sparse(P)  # translate to dict readable for partition

    # Find 5 least probable transitions
    # least_prob_tess = find_least_probable(P,5,M)

    if plotting:
        if dim<4:  # the matrix will be too big
            # Visualize unclustered graph
            plot_graph(P_graph,False,type)
            # Visualize probability matrix
            plot_prob_matrix(P.toarray())

    # plt.show()

    # Clustering--
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
        plot_graph(P_graph_old,True, type)

        # color palette - linear blue
        palette = plt.get_cmap('viridis', D_nodes_in_clusters.shape[1])

    # translate datapoints to cluster number affiliation
    tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

    # cluster centers
    coord_clust_centers, coord_clust_centers_tess = cluster_centers(x,tess_ind, tess_ind_cluster, D_nodes_in_clusters,dim)

    # identify extreme clusters and those transitioning to them
    extr_clusters, from_clusters = extr_iden(extr_trans, D_nodes_in_clusters, P_old)
    print('From cluster: ', from_clusters, 'To extreme cluster: ', extr_clusters)

    for i in range(P_old.shape[0]): # for all unique rows of the deflated probability matrix
        denom = np.sum(D_nodes_in_clusters,axis=0)
        denom = denom[0,i]  # sum of nodes in cluster - we should divide by this
        P_old.data[P_old.row == i] = P_old.data[P_old.row == i]/denom

    if plotting:
        # Plot time series with clusters
        if type=='MFE_burst':
            plot_time_series_clustered(x[:,2], t, tess_ind_cluster, palette, type)
        if type=='MFE_dissipation' or type=='sine'or type=='LA' or type=='CDV' or type=='kolmogorov':
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
        avg_time, nr_instances = avg_time_in_cluster(i,tess_ind_cluster,t)

        clusters.append(cluster(i, nodes, center_coord, center_coord_tess, avg_time, nr_instances, P_old, extr_clusters, from_clusters))

    # for obj in clusters:
    #     print(obj.nr, obj.nodes, obj.center, obj.is_extreme)
    return clusters, D_nodes_in_clusters, P_old      # returns list of clusters (class:cluster), matrix of cluster affiliation and deflated probability matrix (sparse)