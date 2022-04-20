# clustering and extreme event detection for the Pomeau and Manneville equations - 5 dimensional
# Urszula Golyska 2022

import numpy as np
import matplotlib.pyplot as plt
from my_func import *
import seaborn as sns
import networkx as nx
from modularity_maximization import spectralopt
import scipy.sparse as sp

def PM_dt(x):
    # x[:,0] = np.zeros_like(x[:,0])
    # x[:, 1] = np.zeros_like(x[:, 1])

    dxdt = np.zeros(x.shape)
    dxdt[:, 0] = x[:,1]
    dxdt[:, 1] = -x[:,0]**3 - 2*x[:,0]*x[:,2] + x[:,0]*x[:,4] - mu*x[:,1]
    dxdt[:, 2] = x[:,3]
    dxdt[:, 3] = -x[:,2]**3-nu[0]*x[:,0]**2 + x[:,4]*x[:,2] - nu[1]*x[:,3]
    dxdt[:, 4] = -nu[2]*x[:,4] - nu[3]*x[:,0]**2 - nu[4] *(x[:,2]**2 - 1)

    return dxdt

plt.close('all') # close all open figures

# model coefficients
mu = 1.815
nu = [1.0, 1.815, 0.44, 2.86, 2.86]     # to restore the skew product structure [0, 1.815, 0.44, 0, 2.86]

t0 = 0.0
tf = 1000.0
dt = 0.1
t = np.arange(t0,tf,dt)
N = np.size(t)
dim = 5 # number of dimensions

x = np.zeros((N,dim))
x[0,:] = np.array([1,1,0,0,0]) # initial condition

for i in range(N-1):
    q = PM_dt(x[i:i+1,:])
    x[i+1,:] = x[i,:] + dt*q[0,:]

# cutoff first couple timesteps
x = x[500:,:]
t = t[500:]

# Visualize dataset
fig, axs = plt.subplots(3)
fig.suptitle("Dynamic behavior of the dimensions of a PM flow")

plt.subplot(3,1,1)
plt.plot(t, x[:,0])
plt.ylabel("$x_1$")
plt.xlabel("t")
plt.subplot(3,1,2)
plt.plot(t, x[:,1])
plt.ylabel("$x_2$")
plt.xlabel("t")
plt.subplot(3,1,3)
plt.plot(t, x[:,2])
plt.ylabel("$x_3$")
plt.xlabel("t")

fig, axs = plt.subplots(2)
fig.suptitle("Dynamic behavior of the dimensions of a PM flow")

plt.subplot(2,1,1)
plt.plot(t, x[:,3])
plt.ylabel("$x_4$")
plt.xlabel("t")
plt.subplot(2,1,2)
plt.plot(t, x[:,4])
plt.ylabel("$x_5$")
plt.xlabel("t")

# x3 and x5 space
plt.figure(figsize=(6,6))
plt.plot(x[:,2], x[:,4])
plt.ylabel("$x_5$")
plt.xlabel("$x_3$")

# plt.show()

# Tesselation
M = 20  # number of divisions for tesselation, uniform for all dimensions
tess_ind, extr_id = tesselate(x,M,0)    # where 0 indicates the dimension by which the extreme event should be identified - here doesn't matter because we don't have an extreeme event

# Transition probability
P = probability(tess_ind, 'classic') # create sparse transition probability matrix

tess_ind_trans = tess_to_lexi(tess_ind,M,dim)
P, extr_trans = prob_to_sparse(P,M, extr_id) # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

# Graph form
P_graph = to_graph_sparse(P)     # translate to dict readable for partition function

# # Visualize unclustered graph
# plt.figure()
# nx.draw(P_graph,with_labels=True)
# # plt.show()

# Clustering
P_community = spectralopt.partition(P_graph, refine=False) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
D_sparse = community_aff_sparse(0, P_community, M, dim, 'first', 1) # matrix of point-to-cluster affiliation

# Deflate the Markov matrix
P1 = sp.coo_matrix((D_sparse.transpose()*P)*D_sparse)
print(np.sum(P1,axis=0).tolist()) # should be approx.(up to rounding errors) equal to number of nodes in each cluster

# Graph form
P1_graph = to_graph(P1.toarray())

# more iterations
P_community_old = P_community
P_old = P1
P_graph_old = P1_graph
D_nodes_in_clusters= D_sparse
int_id=0

while int(np.size(np.unique(np.array(list(P_community_old.values())))))>25 and int_id<10: # condition
    int_id=int_id+1
    P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old, P_old, D_nodes_in_clusters)
    # print(np.sum(D_nodes_in_clusters,axis=0).tolist())

# Visualize clustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph_old,with_labels=True)

# color palette
palette = sns.color_palette(None, D_nodes_in_clusters.shape[1])

# translate datapoints to cluster number affiliation
tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

fig, axs = plt.subplots(3)
fig.suptitle("Dynamic behavior of the dimensions of a PM flow")

plt.subplot(3,1,1)
plt.plot(t, x[:,0])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_1$")
plt.xlabel("t")
plt.subplot(3,1,2)
plt.plot(t, x[:,1])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_2$")
plt.xlabel("t")
plt.subplot(3,1,3)
plt.plot(t, x[:,2])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        loc_col = palette[tess_ind_cluster[i]]
        plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
        plt.text(t[i], 1, str(tess_ind_cluster[i]), rotation=90)  # numbers of clusters
plt.ylabel("$x_3$")
plt.xlabel("t")

fig, axs = plt.subplots(2)
fig.suptitle("Dynamic behavior of the dimensions of a PM flow")

plt.subplot(2,1,1)
plt.plot(t, x[:,3])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_4$")
plt.xlabel("t")
plt.subplot(2,1,2)
plt.plot(t, x[:,4])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        loc_col = palette[tess_ind_cluster[i]]
        plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
        plt.text(t[i], 3, str(tess_ind_cluster[i]), rotation=90)  # numbers of clusters
plt.ylabel("$x_5$")
plt.xlabel("t")

# phase space
plt.figure(figsize=(6,6))
for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
    plt.scatter(x[tess_ind_cluster==i,2], x[tess_ind_cluster==i,4])  # I should relate somehow s to N and the fig size
    x_mean = np.mean(x[tess_ind_cluster == i, 2])
    y_mean = np.mean(x[tess_ind_cluster == i, 4])
    plt.text(x_mean, y_mean, str(i))  # numbers of clusters
plt.ylabel("$x_5$")
plt.xlabel("$x_3$")

plt.show()