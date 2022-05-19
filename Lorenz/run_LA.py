# script for generating flow data for the Lorenz attractor model for Rayleigh-Bernard convection - 3 dimensional
# Urszula Golyska 2022

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from my_func import *
from modularity_maximization import spectralopt
import networkx as nx
import scipy.sparse as sp

def LA_dt(x): # explicit Euler scheme
    dxdt = np.zeros(x.shape)
    dxdt[0] = sigma*(x[1] - x[0])
    dxdt[1] = x[0]*(r-x[2])-x[1]
    dxdt[2] = x[0]*x[1]-b*x[2]
    return dxdt

plt.close('all') # close all open figures

# model coefficients - same as in Kaiser et al. 2014
sigma = 10
b = 8/3
r = 28

# time discretization
t0 = 0.0
tf = 110.0
dt = 0.01
t = np.arange(t0,tf,dt)
N = np.size(t)

dim=3   # three dimensional
x = np.zeros((N,dim))
x[0,:] = np.array([0, 1.0, 1.05]) # initial condition

for i in range(N-1):
    q = LA_dt(x[i,:])
    x[i+1,:] = x[i,:] + dt*q

# delete first 500 ts to avoid numerical instabilities
x = x[500:,:]
t = t[500:]

# Visualize data
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Lorenz Attractor")

# Visualization of the two dimensions - time series
fig, axs = plt.subplots(3)
# fig.suptitle('Vertically stacked subplots')

plt.subplot(3,1,1)
plt.plot(t, x[:,0])
plt.ylabel("$x$")
plt.xlim((t0,tf))
plt.xlabel("t")
plt.subplot(3,1,2)
plt.plot(t, x[:,1])
plt.ylabel("$y$")
plt.xlim(t0,tf)
plt.xlabel("t")
plt.subplot(3,1,3)
plt.plot(t, x[:,2])
plt.ylabel("$z$")
plt.xlim(t0,tf)
plt.xlabel("t")

# Tesselation
M = 20  # number of divisions for tesselation, uniform for all dimensions
tess_ind, extr_id = tesselate(x,M,0)    # where 0 indicates the dimension by which the extreme event should be identified - here doesn't matter because we don't have an extreeme event

# Visualize tessellation
ax = plt.figure().add_subplot(projection='3d')
ax.scatter3D(tess_ind[:,0], tess_ind[:,1], tess_ind[:,2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Transition probability
P = probability(tess_ind, 'classic') # create sparse transition probability matrix
# P = prob_classic(tess_ind)

tess_ind_trans = tess_to_lexi(tess_ind,M,dim)
P, extr_trans = prob_to_sparse(P,M, extr_id) # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

# Graph form
P_graph = to_graph_sparse(P)     # translate to dict readable for partition function

# Visualize unclustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph,with_labels=True)

# Clustering
P_community = spectralopt.partition(P_graph, refine=False) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
D_sparse = community_aff_sparse(0, P_community, M, dim, 'first', 1)

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

while int(np.size(np.unique(np.array(list(P_community_old.values())))))>15 and int_id<10: # condition
    int_id=int_id+1
    P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old, P_old, D_nodes_in_clusters)

# Visualize clustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph_old,with_labels=True)

# color palette
palette = sns.color_palette(None, D_nodes_in_clusters.shape[1])

# translate datapoints to cluster number affiliation
tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

# Color tesselation hypercubes by cluster affiliation
plt.figure()
ax = plt.axes(projection='3d')
for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
    ax.scatter3D(tess_ind[tess_ind_cluster==i,0], tess_ind[tess_ind_cluster==i,1], tess_ind[tess_ind_cluster==i,2])  # I should relate somehow s to N and the fig size
    x_mean = np.mean(tess_ind[tess_ind_cluster==i,0])
    y_mean = np.mean(tess_ind[tess_ind_cluster==i,1])
    z_mean = np.mean(tess_ind[tess_ind_cluster==i,2])
    ax.text(x_mean, y_mean, z_mean, str(i))  # numbers of clusters
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


plt.figure()
plt.subplot(3,1,1)
plt.plot(t, x[:,0])
# add dashed red line for cluster change
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        loc_col = palette[tess_ind_cluster[i]]
        plt.axvline(x=(t[i]+t[i+1])/2, color = loc_col, linestyle = '--')
        plt.text(t[i], 3, str(tess_ind_cluster[i]), rotation=90)  # numbers of clusters
plt.ylabel("$x$")
plt.xlabel("t")
plt.subplot(3,1,2)
plt.plot(t, x[:,1])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$y$")
plt.xlabel("t")
plt.subplot(3,1,3)
plt.plot(t, x[:,2])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$z$")
plt.xlabel("t")

plt.show()