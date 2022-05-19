# testing the created tesselation and transition probability matrix functions on a simple sine wave case
# Urszula Golyska 2022
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from my_func import *
from modularity_maximization import spectralopt
import scipy.sparse as sp
import seaborn as sns

plt.close('all') # close all open figures

# time discretization
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex = 50  # number of time steps of the extreme event
t = np.arange(t0,tf,dt)

# phase space
u1 = np.sin(t)
u2 = np.cos(t)

# number of dimensions
dim = 2

# generate random spurs
for i in range(len(t)):
    if u1[i]>=0.99 and abs(u2[i])<=0.015:
        if np.random.rand() >=0.9:
            u1[i:i+nt_ex] = u1[i:i+nt_ex]+2
            u2[i:i+nt_ex] = u2[i:i+nt_ex]+2

u = np.hstack([np.reshape(u1, (len(t),1)),np.reshape(u2, (len(t),1))])  # combine into one matrix


# Visualization of the two dimensions - time series
fig, axs = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
plt.subplot(2,1,1)
plt.plot(t, u1)
plt.xlim(t0,tf)
plt.ylabel("$x$")
plt.xlabel("t")
plt.subplot(2,1,2)
plt.plot(t, u2)
plt.ylabel("$y$")
plt.xlim(t0,tf)
plt.xlabel("t")

# Phase space u1,u2
plt.figure(figsize=(7, 7))
plt.plot(u1,u2)
plt.grid('minor', 'both')
plt.minorticks_on()
plt.xlabel("$x$")
plt.ylabel("$y$")

# Tesselation
N = 20  # number of discretizations in each dimension
tess_ind, extr_id = tesselate(u,N,0)    # where 0 indicates the dimension by which the extreme event should be identified - here u1

# Visualization - tesselated space
plt.figure(figsize=(7, 7))
plt.scatter(tess_ind[:,0], tess_ind[:,1], s=200, marker='s', facecolors = 'None', edgecolor='C0') #I should relate somehow s to N and the fig size
plt.grid('minor', 'both')
plt.minorticks_on()
plt.xlabel("$x$")
plt.ylabel("$y$")

# Transition probability matrix
# break up process- first translate to 2d matrix, then count the probabilities, because we need the translated stated wrt t at the end
P = probability(tess_ind, 'classic') # create sparse transition probability matrix
# P = prob_classic(tess_ind)

tess_ind_trans = tess_to_lexi(tess_ind,N, dim)
P, extr_trans = prob_to_sparse(P,N, extr_id) # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

# Visualize probability matrix
plt.figure(figsize=(7, 7))
plt.imshow(P.toarray(),interpolation='none', cmap='binary')
plt.colorbar()

# Graph form
P_graph = to_graph_sparse(P)     # translate to dict readable for partition function

# Visualize unclustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph,with_labels=True)

# Clustering
P_community = spectralopt.partition(P_graph) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
D_sparse = community_aff_sparse(0, P_community, N, dim, 'first', 1) # matrix of point-to-cluster affiliation

# Deflate the Markov matrix
P1 = sp.coo_matrix((D_sparse.transpose()*P)*D_sparse)
print(np.sum(P1,axis=0).tolist()) # should be approx.(up to rounding errors) equal to number of nodes in each cluster

# Graph form
P1_graph = to_graph_sparse(P1)

# more iterations
P_community_old = P_community
P_old = P1
P_graph_old = P1_graph
D_nodes_in_clusters= D_sparse
int_id=0

while int(np.size(np.unique(np.array(list(P_community_old.values())))))>10 and int_id<1: # condition
    int_id=int_id+1
    P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old, P_old, D_nodes_in_clusters)
    # print(np.sum(D_nodes_in_clusters,axis=0).tolist())

# Visualize clustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph_old,with_labels=True)

# extreme_from, extreme_to, nodes_from, nodes_to = extr_iden(P_old,P_community, 'deviation', extr_trans) # for bifurcation it detect's wrong path
extr_cluster,from_cluster = extr_iden(extr_trans, D_nodes_in_clusters, P_old)
print('From cluster: ', from_cluster, 'To extreme cluster: ', extr_cluster)

# translate datapoints to cluster number affiliation
tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

# color palette
palette = sns.color_palette(None, D_nodes_in_clusters.shape[1])
# palette = rgb2hex(palette)

# Identify where we have a from node followed by a to node
# Visualization
fig, axs = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
plt.subplot(2,1,1)
plt.plot(t, u1)
plt.ylabel("$x$")
plt.xlim(t0,tf)
plt.xlabel("t")
plt.subplot(2,1,2)
plt.plot(t, u2)
plt.ylabel("$y$")
plt.xlabel("t")
plt.xlim(t0,tf)
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i] in from_cluster and tess_ind_cluster[i+1] in extr_cluster:
        axs[0].scatter(t[i], u1[i], marker='s', facecolors = 'None', edgecolor = 'blue')
        axs[0].scatter(t[i+1], u1[i+1], marker='s', facecolors='None', edgecolor='red')
        axs[1].scatter(t[i], u2[i], marker='s', facecolors='None', edgecolor='blue')
        axs[1].scatter(t[i+1], u2[i+1], marker='s', facecolors='None', edgecolor='red')


plt.figure(figsize=(13,6))
plt.subplot(2,1,1)
plt.plot(t, u[:,0])
# add dashed red line for cluster change
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        # color associated with previous cluster (transitioning from)
        loc_col = palette[tess_ind_cluster[i]]
        plt.axvline(x=(t[i]+t[i+1])/2, linestyle = '--', color = loc_col)
        plt.text(t[i], 1, str(tess_ind_cluster[i]), rotation=90)    # numbers of clusters
plt.ylabel("$x$")
plt.xlabel("t")
plt.subplot(2,1,2)
plt.plot(t, u[:,1])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, linestyle = '--')
plt.ylabel("$y$")
plt.xlabel("t")

# phase space
plt.figure(figsize=(7, 7))
plt.plot(u1,u2, 'black','-.')
for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
    plt.scatter(u1[tess_ind_cluster==i], u2[tess_ind_cluster==i])#, c=palette[i])  # I should relate somehow s to N and the fig size
    x_mean = np.mean(u1[tess_ind_cluster==i])
    y_mean =np.mean(u2[tess_ind_cluster==i])
    plt.text(x_mean, y_mean, str(i))  # numbers of clusters
plt.grid('minor', 'both')
plt.minorticks_on()
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.show()