# function for running the algorithm for the MFE system
# Urszula Golyska 2022

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import h5py
from my_func import *
import networkx as nx
from modularity_maximization import spectralopt
import scipy.sparse as sp
import seaborn as sns

def data_generation():
    return 1

def read_Fourier(filename):
    hf = h5py.File(filename, 'r')
    u = np.array(hf.get('/u'))
    t = np.array(hf.get('/t'))
    return t,u

def to_burst(u):
    mean_shear = np.absolute(1-u[:,0])    # from a1,a9
    roll_streak = linalg.norm(u[:,1:3].transpose(),axis=0)  # from a2,a3,a4
    burst = linalg.norm(u[:,3:5].transpose(),axis=0)    # from a5,a6,a7,a8
    x = np.vstack([roll_streak, mean_shear, burst]).transpose()
    return x

def read_DI(filename, dt):
    D = np.load(filename+'_dissipation.npy')
    I = np.load(filename+'_energy.npy')
    t = np.arange(len(I))*dt
    x = np.append(D, I, axis=1)
    return t,x

def plot_time_series(x,t, type):
    if type=='burst':
        fig, axs = plt.subplots(5)
        fig.suptitle('Vertically stacked subplots')
        plt.subplot(5,1,1)
        plt.plot(t,u[:,0])
        plt.ylabel("$a_1$")
        plt.xlabel("t")
        plt.subplot(5,1,2)
        plt.plot(t,u[:,1])
        plt.ylabel("$a_2$")
        plt.xlabel("t")
        plt.subplot(5,1,3)
        plt.plot(t,u[:,2])
        plt.ylabel("$a_3$")
        plt.xlabel("t")
        plt.subplot(5,1,4)
        plt.plot(t,u[:,3])
        plt.ylabel("$a_4$")
        plt.xlabel("t")
        plt.subplot(5,1,5)
        plt.plot(t,u[:,4])
        plt.ylabel("$a_5$")
        plt.xlabel("t")

        fig, axs = plt.subplots(4)
        fig.suptitle('Vertically stacked subplots')
        plt.subplot(4,1,1)
        plt.plot(t,u[:,5])
        plt.ylabel("$a_6$")
        plt.xlabel("t")
        plt.subplot(4,1,2)
        plt.plot(t,u[:,6])
        plt.ylabel("$a_7$")
        plt.xlabel("t")
        plt.subplot(4,1,3)
        plt.plot(t,u[:,7])
        plt.ylabel("$a_8$")
        plt.xlabel("t")
        plt.subplot(4,1,4)
        plt.plot(t,u[:,8])
        plt.ylabel("$a_9$")
        plt.xlabel("t")

    if type=='dissipation':
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        plt.subplot(2, 1, 1)
        plt.plot(t, x[:, 0])
        plt.ylabel("D")
        plt.xlabel("t")
        plt.subplot(2, 1, 2)
        plt.plot(t, x[:, 1])
        plt.ylabel("I")
        plt.xlabel("t")

    return 1

def plot_phase_space(x, type):
    if type=='burst':
        plt.figure()
        plt.plot(t, x[2,:])
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
        plt.xlabel("t")

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[0,:], x[1,:], x[2,:], lw=0.5)
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='dissipation':
        plt.figure()
        plt.plot(I, D)
        plt.title("Dissipation vs energy")
        plt.ylabel("D")
        plt.xlabel("I")
    return 1

def plot_tesselated_space(tess_ind,type):
    if type=='burst':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:, 0], tess_ind[:, 1], tess_ind[:, 2])
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")

    if type=='dissipation':
        plt.figure(figsize=(7, 7))
        plt.scatter(tess_ind[:,0], tess_ind[:,1], s=200, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("I")
        plt.ylabel("D")
    return 1

def plot_prob_matrix(P_dense):
    # Visualize probability matrix
    plt.figure(figsize=(7, 7))
    plt.imshow(P_dense,interpolation='none', cmap='binary')
    plt.colorbar()
    return 1

def plot_graph(P_graph):
    # Visualize unclustered graph
    plt.figure()
    nx.draw_kamada_kawai(P_graph,with_labels=True)
    return 1

def plt_phase_space_clustered(x,type,D_nodes_in_clusters,tess_ind_cluster ):
    if type=='burst':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2])  # I should relate somehow s to N and the fig size
            x_mean = np.mean(x[tess_ind_cluster == i, 0])
            y_mean = np.mean(x[tess_ind_cluster == i, 1])
            z_mean = np.mean(x[tess_ind_cluster == i, 2])
            ax.text(x_mean, y_mean, z_mean, str(i))  # numbers of clusters
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='dissipation':
        plt.figure(figsize=(7, 7))
        # plt.plot(I,D, 'b--')
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i,1],
                        x[tess_ind_cluster == i,0])  # , c=palette[i])  # I should relate somehow s to N and the fig size
            x_mean = np.mean(x[tess_ind_cluster == i,1])
            y_mean = np.mean(x[tess_ind_cluster == i,0])
            plt.text(x_mean, y_mean, str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("I")
        plt.ylabel("D")

    return 1

def plot_time_series_clustered(y,t, tess_ind_cluster, palette, type): #type=burst -> y=burst; type=diss -> y=D
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
            loc_col = palette[tess_ind_cluster[i]]
            plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
            plt.text(t[i], 0.2, str(tess_ind_cluster[i]), rotation=90)  # numbers of clusters
    if type=='burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='dissipation':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    plt.xlabel("t")

    return 1

def plot_time_series_extr_iden(y,t, tess_ind_cluster, from_cluster, extr_cluster, type):
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i] in from_cluster and tess_ind_cluster[i+1] in extr_cluster:
            plt.scatter(t[i], y[i], marker='s', facecolors = 'None', edgecolor = 'blue')
            plt.scatter(t[i+1], y[i+1], marker='s', facecolors='None', edgecolor='red')
    if type=='burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='dissipation':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    plt.xlabel("t")
    return 1

def MFE_process(t,x,dim,M,extr_dim,type):
    tess_ind, extr_id = tesselate(x, M, extr_dim)  # where extr_dim indicates the dimension by which the extreme event should be identified
    # Transition probability
    P = probability(tess_ind, 'classic')  # create sparse transition probability matrix

    tess_ind_trans = tess_to_lexi(tess_ind, M, dim)
    P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

    # Graph form
    P_graph = to_graph_sparse(P)  # translate to dict readable for partition

    # Visualize unclustered graph
    plot_graph(P_graph)

    # Visualize probability matrix
    plot_prob_matrix(P.toarray())

    # Clustering
    P_community = spectralopt.partition(P_graph, refine=False)  # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
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

    while int(np.size(np.unique(np.array(list(P_community_old.values()))))) > 20 and int_id < 10:  # condition
        int_id = int_id + 1
        P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old,
                                                                                          P_old, D_nodes_in_clusters)
        # print(np.sum(D_nodes_in_clusters,axis=0).tolist())

    # Visualize clustered graph
    plot_graph(P_graph_old)

    # color palette
    palette = sns.color_palette(None, D_nodes_in_clusters.shape[1])

    # translate datapoints to cluster number affiliation
    tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

    # identify extreme clusters and those transitioning to them
    extr_cluster, from_cluster = extr_iden(extr_trans, D_nodes_in_clusters, P_old)
    print('From cluster: ', from_cluster, 'To extreme cluster: ', extr_cluster)

    # Plot time series with clusters
    if type=='burst':
        plot_time_series_clustered(x[:,2], t, tess_ind_cluster, palette, type)
    if type=='dissipation':
        plot_time_series_clustered(x[:,0], t, tess_ind_cluster, palette, type)

    # Visualize phase space trajectory with clusters
    plt_phase_space_clustered(x, type, D_nodes_in_clusters, tess_ind_cluster)

    # Plot time series with extreme event identification
    if type == 'burst':
        plot_time_series_extr_iden(x[:,2], t, tess_ind_cluster, from_cluster, extr_cluster, type)
    if type == 'dissipation':
        plot_time_series_extr_iden(x[:,0], t, tess_ind_cluster, from_cluster, extr_cluster, type)

    # Visualize probability matrix
    plot_prob_matrix(P_old.toarray())

    return 1