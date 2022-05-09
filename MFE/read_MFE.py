import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.integrate import ode
import h5py
from my_func import *
import networkx as nx
from modularity_maximization import spectralopt
import scipy.sparse as sp
import seaborn as sns

plt.close('all') # close all open figures

hf = h5py.File('MFE_Re600_DATA.h5','r')
# hf = h5py.File('MFE_Re600_DATA_dissipation.h5','r')
# D = np.array(hf.get('/D'))

# plt.plot(range(10000),D[:10000])
# plt.show()


Lx = np.array(hf.get('/Lx'))
Lz = np.array(hf.get('/Lz'))
Re = np.array(hf.get('/Re'))
alpha = np.array(hf.get('/alpha'))
beta = np.array(hf.get('/beta'))
dt = np.array(hf.get('/dt'))
gamma = np.array(hf.get('/gamma'))
u0 = np.array(hf.get('/u0'))
xi1 = np.array(hf.get('/xi1'))
xi2 = np.array(hf.get('/xi2'))
xi3 = np.array(hf.get('/xi3'))
xi4 = np.array(hf.get('/xi4'))
xi5 = np.array(hf.get('/xi5'))
xi6 = np.array(hf.get('/xi6'))
xi7 = np.array(hf.get('/xi7'))
xi8 = np.array(hf.get('/xi8'))
xi9 = np.array(hf.get('/xi9'))
zeta = np.array(hf.get('/zeta'))
u = np.array(hf.get('/u'))
t = np.array(hf.get('/t'))

# delete initial period
t = t[500:]
u = u[500:, :]

# Visualize coefficients
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

# D =
# I =

mean_shear = np.absolute(1-u[:,0])    # from a1,a9
roll_streak = linalg.norm(u[:,1:3].transpose(),axis=0)  # from a2,a3,a4
burst = linalg.norm(u[:,3:5].transpose(),axis=0)    # from a5,a6,a7,a8

m = np.mean(burst)
print(m)
# temp = abs(burst-m)/max(abs(burst-m))
# extr_id = tess_ind[temp>=0.9,:]
cutoff = m+0.7*(max(burst)-m)

plt.figure()
plt.plot(t, burst)
plt.axhline(cutoff, color='r', linestyle='-.')
plt.title("Burst component vs time")
plt.ylabel("$b$")
plt.xlabel("t")

plt.show()

# Visualize data
ax = plt.figure().add_subplot(projection='3d')
ax.plot(roll_streak, mean_shear, burst, lw=0.5)
ax.set_xlabel("Roll & streak")
ax.set_ylabel("Mean shear")
ax.set_zlabel("Burst")
ax.set_title("Self-sustaining process")

x = np.vstack([roll_streak, mean_shear, burst]).transpose()
dim = 3 # number of dimensions

# Tesselation
M = 20  # number of divisions for tesselation, uniform for all dimensions
tess_ind, extr_id = tesselate(x,M,2)    # where 2 indicates the dimension by which the extreme event should be identified - here burst

# Transition probability
P = probability(tess_ind, 'classic') # create sparse transition probability matrix
#
tess_ind_trans = tess_to_lexi(tess_ind,M,dim)
P, extr_trans = prob_to_sparse(P,M, extr_id) # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

# Graph form
P_graph = to_graph_sparse(P)     # translate to dict readable for partition function

# Visualize unclustered graph
plt.figure()
nx.draw(P_graph,with_labels=True)

# Visualize probability matrix
plt.figure(figsize=(7, 7))
plt.imshow(P.toarray(),interpolation='none', cmap='binary')
plt.colorbar()

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

while int(np.size(np.unique(np.array(list(P_community_old.values())))))>20 and int_id<10: # condition
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

# plt.figure()
# plt.plot(t, burst)
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         loc_col = palette[tess_ind_cluster[i]]
#         plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
#         plt.text(t[i], 0.2, str(tess_ind_cluster[i]), rotation=90)  # numbers of clusters
# plt.title("Burst component vs time")
# plt.ylabel("$b$")
# plt.xlabel("t")

plt.figure()
plt.plot(t, burst)

cluster_start=tess_ind_cluster[0]
i_start = 0
for i in range(len(tess_ind_cluster)):
    if tess_ind_cluster[i]!=cluster_start:
        loc_col = palette[cluster_start]
        plt.plot(t[i_start:i-1], burst[i_start:i-1],color = loc_col)
        cluster_start = tess_ind_cluster[i]
        i_start = i
plt.title("Burst component vs time")
plt.ylabel("$b$")
plt.xlabel("t")

# phase space
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

# extreme_from, extreme_to, nodes_from, nodes_to = extr_iden(P_old,P_community, 'deviation', extr_trans) # for bifurcation it detect's wrong path
extr_cluster,from_cluster = extr_iden(extr_trans, D_nodes_in_clusters, P_old)
print('From cluster: ', from_cluster, 'To extreme cluster: ', extr_cluster)

plt.figure()
plt.plot(t, burst)
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i] in from_cluster and tess_ind_cluster[i+1] in extr_cluster:
        plt.scatter(t[i], burst[i], marker='s', facecolors = 'None', edgecolor = 'blue')
        plt.scatter(t[i+1], burst[i+1], marker='s', facecolors='None', edgecolor='red')
plt.title("Burst component vs time")
plt.ylabel("$b$")
plt.xlabel("t")


# Visualize probability matrix
plt.figure(figsize=(7, 7))
plt.imshow(P_old.toarray(),interpolation='none', cmap='binary')
plt.colorbar()

plt.show()
