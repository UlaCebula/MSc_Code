# testing the created tesselation and transition probability matrix functions on a simple circle case
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from my_func import *
from modularity_maximization import spectralopt

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

# # Visualization of the two dimensions - time series
# fig, axs = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# axs[0].plot(t, u1)
# plt.ylabel("$u_1$")
# plt.xlabel("t")
# axs[1].plot(t, u2)
# plt.ylabel("$u_2$")
# plt.xlabel("t")

# # Phase space u1,u2
# plt.figure(figsize=(7, 7))
# plt.plot(u1,u2)
# plt.grid('minor', 'both')
# plt.minorticks_on()
# plt.xlabel("$u_1$")
# plt.ylabel("$u_2$")

# Tesselation
N = 20  # number of discretizations in each dimension
tess_ind, extr_id = tesselate(u,N,0)    # where 0 indicates the dimension by which the extreme event should be identified - here u1

# # Visualization - tesselated space
# plt.figure(figsize=(7, 7))
# plt.scatter(tess_ind[:,0], tess_ind[:,1], s=200, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
# plt.grid('minor', 'both')
# plt.minorticks_on()
# plt.xlabel("$u_1$")
# plt.ylabel("$u_2$")

# Transition probability matrix
# break up process- first translate to 2d matrix, then count the probabilities, because we need the translated stated wrt t at the end
P = probability(tess_ind, 'backwards') # create sparse transition probability matrix
print(tess_ind.max())

tess_ind_trans = tess_to_lexi(tess_ind,N, dim)
P, extr_trans = prob_to_sparse(P,N, extr_id) # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id
print(tess_ind.max())

P_dense = P.toarray()
# A=P_dense.sum(axis=0) # for the classic approach, must have prob=1 for all the from points

# # Visualize probability matrix
# plt.figure(figsize=(7, 7))
# plt.imshow(P_dense,interpolation='none', cmap='binary')
# plt.colorbar()

# Graph form
P_graph = to_graph(P_dense)     # translate to dict readable for partition function

# # Visualize unclustered graph
# plt.figure()
# nx.draw_kamada_kawai(P_graph,with_labels=True)

# Clustering
P_community = spectralopt.partition(P_graph) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
D = community_aff(P_community, N, dim, 1) # matrix of point-to-cluster affiliation

# Deflate the Markov matrix
P1 = np.matmul(np.matmul(D.transpose(),P_dense.transpose()),D)
print(np.sum(P1,axis=0).tolist()) # should be approx.(up to rounding errors) equal to number of nodes in each cluster

# Graph form
P1 = P1.transpose()   # had to add transpose for the classic probability, why? the same for backwards?
P1_graph = to_graph(P1)

# Visualize clustered graph
# plt.figure()
# nx.draw_kamada_kawai(P1_graph,with_labels=True)

# Identify extreme event it's precursor
extreme_from, extreme_to, nodes_from, nodes_to = extr_iden(P1,P_community, 'deviation', extr_trans) # for bifurcation it detect's wrong path
print('From: ', extreme_from,'(',nodes_from,') \n', 'To: ', extreme_to, '(', nodes_to, ')')


# Identify where we have a from node followed by a to node
# Visualization
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(t, u1)
plt.ylabel("$u_1$")
plt.xlabel("t")
axs[1].plot(t, u2)
plt.ylabel("$u_2$")
plt.xlabel("t")

# for i in range(len(tess_ind_trans)-1):
#     if tess_ind_trans[i] in nodes_from:
#         # t_from = t0 + dt * i
#         # axs[0].scatter(t_from, u1[i], marker='s', facecolors='None', edgecolor='blue')
#         if tess_ind_trans[i+1] in nodes_to:
#             t_from = t0 + dt*i
#             t_to = t_from + dt
#             axs[0].scatter(t_from, u1[i], marker='s', facecolors = 'None', edgecolor = 'blue')
#             axs[0].scatter(t_to, u1[i+1], marker='s', facecolors='None', edgecolor='red')
#             axs[1].scatter(t_from, u2[i], marker='s', facecolors='None', edgecolor='blue')
#             axs[1].scatter(t_to, u2[i+1], marker='s', facecolors='None', edgecolor='red')

# easier way
temp_from = np.where(np.isin(tess_ind_trans, nodes_from))
axs[0].scatter(t[temp_from], u1[temp_from], marker='s', facecolors = 'None', edgecolor = 'blue')
axs[1].scatter(t[temp_from], u2[temp_from], marker='s', facecolors = 'None', edgecolor = 'blue')
temp_to = np.where(np.isin(tess_ind_trans, nodes_to))
axs[0].scatter(t[temp_to], u1[temp_to], marker='s', facecolors = 'None', edgecolor = 'red')
axs[1].scatter(t[temp_to], u2[temp_to], marker='s', facecolors = 'None', edgecolor = 'red')


plt.show()