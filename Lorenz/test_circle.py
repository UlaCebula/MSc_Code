# testing the created tesselation and transition probability matrix functions on a simple circle case
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from my_func import tesselate, trans_matrix, prob_to_sparse
import modularity_maximization as mm

# time discretization
t0 = 0.0
tf = 100.0
dt = 0.01
t = np.arange(t0,tf,dt)

# phase space
u1 = np.sin(t)
u2 = np.cos(t)

# generate random spurs
for i in range(len(t)):
    if u1[i]>=0.99 and abs(u2[i])<=0.01:
        if np.random.rand() >=0.75:
            u1[i] = 2
            u2[i] = 2

# Visualization of the two dimensions
plt.figure(figsize=(13, 2))
plt.plot(t,u1, '--')
plt.plot(t,u2, '-.')
plt.legend(['$u_1$','$u_2$'], loc="upper left")
plt.grid('minor', 'both')
plt.minorticks_on()
plt.xlabel("t")
plt.ylabel("u")

plt.figure(figsize=(7, 7))
plt.plot(u1,u2)
plt.grid('minor', 'both')
plt.minorticks_on()
plt.xlabel("$u_1$")
plt.ylabel("$u_2$")

# combine into one matrix
u = np.hstack([np.reshape(u1, (len(t),1)),np.reshape(u2, (len(t),1))])

# tesselate
N = 20
tess_ind = tesselate(u,N)

# tesselated space visualization
plt.figure(figsize=(7, 7))
plt.scatter(tess_ind[:,0], tess_ind[:,1], s=100, marker='s', facecolors = 'None', edgecolor = 'blue')
plt.grid('minor', 'both')
plt.minorticks_on()
plt.xlabel("$u_1$")
plt.ylabel("$u_2$")

# transition probability matrix
P = trans_matrix(tess_ind)  # create sparse transition probability matrix
P = prob_to_sparse(P,N) # translate matrix into 2D sparse array with points in lexicographic order

P_dense = P.toarray()
# print(P)

# visualize probability matrix
plt.figure(figsize=(7, 7))
plt.imshow(P_dense,interpolation='none', cmap='binary')
plt.colorbar()
# plt.show()

# clustering
# translate to dict readable for partition function
P_graph = nx.DiGraph()
for i in range(len(P.row)):
    # row = P_coord[i]
    temp=[P.row[i], P.col[i], P.data[i]]
    P_graph.add_edge(P.row[i], P.col[i], weight=P.data[i])
# print(P_graph)

# visualize graph
# plt.figure()
# nx.draw(P_graph,with_labels=True)
# plt.show()

# clustering
P_graph = mm.partition(P_graph) # partition community P, default with refinement

# visualize graph
plt.figure()
nx.draw(P_graph,with_labels=True)
plt.show()
