# testing the created tesselation and transition probability matrix functions on a simple circle case
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from my_func import tesselate, trans_matrix, prob_to_sparse, community_aff, to_graph
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
dim=2

# generate random spurs
for i in range(len(t)):
    if u1[i]>=0.99 and abs(u2[i])<=0.015:
        if np.random.rand() >=0.9:
            u1[i:i+nt_ex] = u1[i:i+nt_ex]+2
            u2[i:i+nt_ex] = u2[i:i+nt_ex]+2

# Visualization of the two dimensions
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(t, u1)
plt.ylabel("$u_1$")
plt.xlabel("t")
axs[1].plot(t, u2)
plt.ylabel("$u_2$")
plt.xlabel("t")


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
plt.scatter(tess_ind[:,0], tess_ind[:,1], s=200, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
plt.grid('minor', 'both')
plt.minorticks_on()
plt.xlabel("$u_1$")
plt.ylabel("$u_2$")

# transition probability matrix
# break up process- first translate to 2d matrix, then count the probabilities, because we need the translated stated wrt t at the end
P = trans_matrix(tess_ind)  # create sparse transition probability matrix
P = prob_to_sparse(P,N) # translate matrix into 2D sparse array with points in lexicographic order

P_dense = P.toarray().transpose()
# print(P)

# visualize probability matrix
plt.figure(figsize=(7, 7))
plt.imshow(P_dense,interpolation='none', cmap='binary')
plt.colorbar()
# plt.show()

# clustering
# translate to dict readable for partition function
P_graph = to_graph(P_dense)

# visualize graph
plt.figure()
nx.draw_kamada_kawai(P_graph,with_labels=True)

# clustering
P_community = spectralopt.partition(P_graph) # partition community P, default with refinement; returns dict where nodes are keys and values are community indices

D = community_aff(P_community, N, dim, 1) # matrix of point-to-cluster affiliation

# now deflate the Markov matrix
P1 = np.matmul(np.matmul(D.transpose(),P_dense),D)
print(np.sum(P1,axis=1).tolist())

# translate to graph
P1_graph = to_graph(P1)

# visualize graph
plt.figure()
nx.draw_kamada_kawai(P1_graph,with_labels=True)

# identify extreme event and transition to it
# rescale probability matrix so that probabilities are one:
# for i in range(len(P1[:,0])):
#     P1[i,:] = P1[i,:]/np.sum(P1[i,:])
# find transition to extreme event
extreme_from = np.where(np.count_nonzero(P1, axis=1)>2)    #will give the row
extreme_from = int(extreme_from[0])
extreme_to = np.where(P1[extreme_from,:]==P1[extreme_from,P1[extreme_from,:].nonzero()].min())    # indentifies clusters from and to which we have the extreme event transition
extreme_to = int(extreme_to[0])
print(extreme_from, extreme_to)

# on the same dataset, but cut shorter
# nodes of cluster from and to - there has to be a better way for this
nodes_from=[]
nodes_to=[]
for key, value in P_community.items():
    if value == extreme_from:
        nodes_from.append(key)
    if value == extreme_to:
        nodes_to.append(key)


# but we first have to translate the tess_inf matrix into the one with node numbers/lexicographic order
 # translate points into lexicographic order - there must be a better way, plus add this as a function and make such tahat it works for all dimensions
tess_ind_trans = np.zeros(len(tess_ind[:,0]))
for i in range(len(tess_ind[:,0])):  # loop through all points
    tess_ind_trans[i]=tess_ind[i,0]+tess_ind[i,1]*N  # first point (to)


# identify where we have a from node followed by a to node
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(t, u1)
plt.ylabel("$u_1$")
plt.xlabel("t")
axs[1].plot(t, u2)
plt.ylabel("$u_2$")
plt.xlabel("t")

for i in range(len(tess_ind_trans)-1):
    if tess_ind_trans[i] in nodes_from:
        # t_from = t0 + dt * i
        # axs[0].scatter(t_from, u1[i], marker='s', facecolors='None', edgecolor='blue')
        if tess_ind_trans[i+1] in nodes_to:
            t_from = t0 + dt*i
            t_to = t_from + dt
            axs[0].scatter(t_from, u1[i], marker='s', facecolors = 'None', edgecolor = 'blue')
            axs[0].scatter(t_to, u1[i+1], marker='s', facecolors='None', edgecolor='red')
            axs[1].scatter(t_from, u2[i], marker='s', facecolors='None', edgecolor='blue')
            axs[1].scatter(t_to, u2[i+1], marker='s', facecolors='None', edgecolor='red')
            print(t_from, t_to)
plt.show()

print(1)