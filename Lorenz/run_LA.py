# script for generating flow data for the Lorenz attractor model for Rayleigh-Bernard convection - 3 dimensional
# Urszula Golyska 2022

import numpy as np
import matplotlib.pyplot as plt
from my_func import tesselate, trans_matrix, prob_to_sparse, to_graph, community_aff
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
tf = 100.0 # runtime around 15 min
dt = 0.01
t = np.arange(t0,tf,dt)
N = np.size(t)

dim=3   # three dimensional
x = np.zeros((N,dim))
x[0,:] = np.array([0, 1.0, 1.05]) # initial condition

for i in range(N-1):
    q = LA_dt(x[i,:])
    x[i+1,:] = x[i,:] + dt*q

# Visualize data
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

# Tesselation
M = 20  # number of divisions for tesselation, uniform for all dimensions
tess_ind = tesselate(x,M)    # output - indices of occupied spaces (sparse matrix)

# # for visualization only - works only for the 3d case
# ax = plt.axes(projection='3d')
# ax.scatter3D(tess_ind[:,0], tess_ind[:,1], tess_ind[:,2])
# plt.show()

# Transition probability
P = trans_matrix(tess_ind)  # create sparse transition probability matrix
# print(np.max(P[:,6]), np.min(P[:,6])) # just checking

P = prob_to_sparse(P,M) # translate matrix into 2D sparse array with points in lexicographic order

P_dense = P.toarray()
# print(P)

# visualize probability matrix
plt.figure(figsize=(7, 7))
plt.imshow(P_dense,interpolation='none', cmap='binary')
plt.colorbar()

# clustering
# translate to dict readable for partition function
P_graph = to_graph(P.toarray())

# visualize graph
plt.figure()
nx.draw_kamada_kawai(P_graph,with_labels=True)

# clustering
P_community = spectralopt.partition(P_graph) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
# print all communities and their node entries
D = community_aff(P_community, M, dim, 1) # matrix of point-to-cluster affiliation

# # now deflate the Markov matrix
P1 = np.matmul(np.matmul(D.transpose(),P_dense),D)
print(np.sum(P1,axis=1).tolist())

# translate to graph
P1_graph = to_graph(P1)

# visualize graph
plt.figure()
nx.draw_kamada_kawai(P1_graph,with_labels=True)
plt.show()

