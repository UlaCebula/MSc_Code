# script for generating flow data for the Lorenz attractor model for Rayleigh-Bernard convection - 3 dimensional
# Urszula Golyska 2022

import numpy as np
import matplotlib.pyplot as plt
from my_func import *
from modularity_maximization import spectralopt
import networkx as nx

def LA_dt(x): # explicit Euler scheme
    dxdt = np.zeros(x.shape)
    dxdt[0] = sigma*(x[1] - x[0])
    dxdt[1] = x[0]*(r-x[2])-x[1]
    dxdt[2] = x[0]*x[1]-b*x[2]
    return dxdt

# model coefficients - same as in Kaiser et al. 2014
sigma = 10
b = 8/3
r = 28

# time discretization
t0 = 0.0
tf = 60.0 #110.0
dt = 0.01
t = np.arange(t0,tf,dt)
N = np.size(t)

dim=3   # three dimensional
x = np.zeros((N,dim))
x[0,:] = np.array([0, 1.0, 1.05]) # initial condition

for i in range(N-1):
    q = LA_dt(x[i,:])
    x[i+1,:] = x[i,:] + dt*q

# delete first 1000 ts to avoid numerical instabilities
x = x[500:,:]
t = t[500:]

# Visualize data
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

# Visualization of the two dimensions - time series
fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')

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

# Tesselation
M = 20  # number of divisions for tesselation, uniform for all dimensions
tess_ind, extr_id = tesselate(x,M,0)    # where 0 indicates the dimension by which the extreme event should be identified - here doesn't matter because we don't have an extreeme event

# Visualize tessellation
ax = plt.figure().add_subplot(projection='3d')
ax.scatter3D(tess_ind[:,0], tess_ind[:,1], tess_ind[:,2])
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

# Transition probability
P = probability(tess_ind, 'backwards') # create sparse transition probability matrix
# P = prob_classic(tess_ind)

tess_ind_trans = tess_to_lexi(tess_ind,M,dim)
P, extr_trans = prob_to_sparse(P,M, extr_id) # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

P_dense = P.toarray()
# A=P_dense.sum(axis=0) # for the classic approach, must have prob=1 for all the from points

# Graph form
P_graph = to_graph(P_dense)     # translate to dict readable for partition function

# # Visualize unclustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph,with_labels=True)

# Clustering
P_community = spectralopt.partition(P_graph) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
D = community_aff(P_community, M, dim, 1) # matrix of point-to-cluster affiliation

# Deflate the Markov matrix
P1 = np.matmul(np.matmul(D.transpose(),P_dense.transpose()),D)
print(np.sum(P1,axis=0).tolist()) # should be approx.(up to rounding errors) equal to number of nodes in each cluster

# Graph form
P1 = P1.transpose()   # had to add transpose for the classic probability, why? the same for backwards?
P1_graph = to_graph(P1)

# more iterations
P_community_old = P_community
P_old = P1
P_graph_old = P1_graph
D_nodes_in_clusters= D

while int(np.size(np.unique(np.array(list(P_community_old.values())))))>22: # while nr communities>20
    P_community_new = spectralopt.partition(P_graph_old) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
    D_new = community_aff_clusters(P_community_old, P_community_new, 1) # matrix of point-to-cluster affiliation

    # Deflate the Markov matrix
    P_new = np.matmul(np.matmul(D_new.transpose(),P_old),D_new) # P1 transposed or not?
    print(np.sum(P_new,axis=0).tolist()) # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P_new = P_new.transpose()   # had to add transpose for the classic probability, why? the same for backwards?
    P_graph_old = to_graph(P_new)
    P_community_old = P_community_new
    P_old = P_new

    # make translation of which nodes belong to the new cluster
    D_nodes_in_clusters = np.matmul(D_nodes_in_clusters,D_new)

# Visualize clustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph_old,with_labels=True)

# Color tesselation hypercubes by cluster affiliation - not efficient!!
plt.figure()
ax = plt.axes(projection='3d')
for i in range(len(D_nodes_in_clusters[0,:])):   # for all communities
    print("Community: ", i)
    print("Nodes: ", end='')
    nodes = np.array(D_nodes_in_clusters[:,i].nonzero())
    print(nodes)
    temp_nodes=[0,0,0]
    for j in range(len(tess_ind_trans)):
        if tess_ind_trans[j] in nodes:
            temp_nodes = np.vstack([temp_nodes, tess_ind[j,:]])
    temp_nodes = temp_nodes[1:,:]
    ax.scatter3D(temp_nodes[:,0], temp_nodes[:,1], temp_nodes[:,2])

plt.show()