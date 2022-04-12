# script for generating flow data for the Lorenz attractor model for Rayleigh-Bernard convection - 3 dimensional
# Urszula Golyska 2022

import matplotlib.pyplot as plt
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
tf = 30.0 #110.0
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

# Color tesselation hypercubes by cluster affiliation - not efficient!!
plt.figure()
ax = plt.axes(projection='3d')
for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
    print("Community: ", i)
    print("Nodes: ", end='')
    nodes = D_nodes_in_clusters.row[D_nodes_in_clusters.col==i]
    print(nodes)
    temp_nodes=[0,0,0]
    for j in range(len(tess_ind_trans)):
        if tess_ind_trans[j] in nodes:
            temp_nodes = np.vstack([temp_nodes, tess_ind[j,:]])
    temp_nodes = temp_nodes[1:,:]
    ax.scatter3D(temp_nodes[:,0], temp_nodes[:,1], temp_nodes[:,2])

plt.show()