# clustering and extreme event detection for the Charney-DeVore equations - 6 dimensional
# Urszula Golyska 2022
# data generation part by Anh Khoa Doan

import numpy as np
import h5py
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from my_func import *
import networkx as nx
from modularity_maximization import spectralopt
import scipy.sparse as sp


def NL(x):

    assert len(x.shape) == 2, 'Input needs to be a two-dimensional array.'
    Nx = np.zeros(x.shape)

    Nx[:,1] = -alpha[0]*x[:,0]*x[:,2] - delta[0]*x[:,3]*x[:,5]
    Nx[:,2] = alpha[0]*x[:,0]*x[:,1] + delta[0]*x[:,3]*x[:,4]
    Nx[:,3] = epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
    Nx[:,4] = -alpha[1]*x[:,0]*x[:,5] - delta[1]*x[:,2]*x[:,3]
    Nx[:,5] = alpha[1]*x[:,0]*x[:,4] + delta[1]*x[:,3]*x[:,1]

    return Nx

def CDV_dt(x):

    #assert len(x.shape) == 2, 'Input needs to be a two-dimensional array.'
    #Nx = np.zeros(x.shape)

    #Nx[:,1] = -alpha[0]*x[:,0]*x[:,2] - delta[0]*x[:,3]*x[:,5]
    #Nx[:,2] = alpha[0]*x[:,0]*x[:,1] + delta[0]*x[:,3]*x[:,4]
    #Nx[:,3] = epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
    #Nx[:,4] = -alpha[1]*x[:,0]*x[:,5] - delta[1]*x[:,2]*x[:,3]
    #Nx[:,5] = alpha[1]*x[:,0]*x[:,4] + delta[1]*x[:,3]*x[:,1]
    
    dxdt = np.zeros(x.shape)
    dxdt[:,0] = gamma_m_star[0]*x[:,2] - C*(x[:,0] - x1s)
    dxdt[:,1] = -(alpha[0]*x[:,0] - beta[0])*x[:,2] - C*x[:,1] - delta[0]*x[:,3]*x[:,5] ## WHY IS THERE A MINUS HERE?
    dxdt[:,2] = (alpha[0]*x[:,0] - beta[0])*x[:,1] - gamma_m[0]*x[:,0] - C*x[:,2] + delta[0]*x[:,3]*x[:,4]
    dxdt[:,3] = gamma_m_star[1]*x[:,5] - C*(x[:,3] - x4s) + epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
    dxdt[:,4] = -(alpha[1]*x[:,0] - beta[1])*x[:,5] - C*x[:,4] - delta[1]*x[:,2]*x[:,3]
    dxdt[:,5] = (alpha[1]*x[:,0] - beta[1])*x[:,4] - gamma_m[1]*x[:,3] - C*x[:,5] + delta[1]*x[:,3]*x[:,1]
    #dxdt = np.matmul(x,L) + NL(x) + b
    return dxdt

plt.close('all') # close all open figures

# chaotic
# model coefficients
x1s = .95
x4s = -.76095
C = .1
beta0 = 1.25
gamma = .2
b = .5

m = np.array([1,2])
alpha = 8*np.sqrt(2)*m**2*(b**2+m**2-1)/np.pi/(4*m**2-1)/(b**2+m**2)
beta = beta0*b**2/(b**2+m**2)
delta = 64*np.sqrt(2)*(b**2-m**2+1)/15/np.pi/(b**2+m**2)
gamma_m = gamma*4*np.sqrt(2)*m**3*b/np.pi/(4*m**2-1)/(b**2+m**2)
gamma_m_star = gamma*4*np.sqrt(2)*m*b/np.pi/(4*m**2-1)
epsilon = 16*np.sqrt(2)/5/np.pi

# linear part of the operator
#L = np.zeros((6,6))
#L[0,0],L[2,0] = -C,gamma_m_star[0]
#L[1,1],L[2,1] = -C,beta[0]
#L[0,2],L[1,2],L[2,2] = -gamma_m[0],-beta[0],-C
#L[3,3],L[5,3] = -C,gamma_m_star[1]
#L[4,4],L[5,4] = -C,beta[1]
#L[3,5],L[4,5],L[5,5] = -gamma_m[1],-beta[1],-C

#b = np.zeros((1,6))
#b[:,0],b[:,3] = C*x1s,C*x4s

t0 = 0.0
tf = 24000.0 #24000.0
dt = 0.1 #0.1
t = np.arange(t0,tf,dt)
N = np.size(t)
dim=6 # number of dimensions

x = np.zeros((N,dim))
x[0,:] = np.array([.11,.22,.33,.44,.55,.66]) # initial condition

for i in range(N-1):
    q = CDV_dt(x[i:i+1,:])
    x[i+1,:] = x[i,:] + dt*q[0,:]

# cutoff first couple timesteps
x = x[500:,:]
t = t[500:]
    
# fln = 'CDV_T' + str(N) + '_DT01.h5'
# hf = h5py.File(fln,'w')
# hf.create_dataset('x',data=x)
# hf.create_dataset('t',data=t)
# hf.create_dataset('dt',data=dt)
# #hf.create_dataset('L',data=L)
# hf.create_dataset('x1s',data=x1s)
# hf.create_dataset('x4s',data=x4s)
# hf.create_dataset('C',data=C)
# hf.create_dataset('beta0',data=beta0)
# hf.create_dataset('gamma',data=gamma)
# #hf.create_dataset('b',data=b)
# hf.create_dataset('m',data=m)
# hf.create_dataset('alpha',data=alpha)
# hf.create_dataset('beta',data=beta)
# hf.create_dataset('delta',data=delta)
# hf.create_dataset('gamma_m',data=gamma_m)
# hf.create_dataset('gamma_m_star',data=gamma_m_star)
# hf.create_dataset('epsilon',data=epsilon)
# hf.close()

# plt.figure(figsize=(13, 7))
# plt.plot(t,x, '--', linewidth=0.75)
# plt.legend(['$x_1$','$x_2$','$x_3$','$x_4$','$x_5$','$x_6$'], loc="upper left", ncol=6)
# plt.grid('minor', 'both')
# plt.minorticks_on()
# plt.title("Dynamic behavior of the dimensions of a CDV flow")
# plt.xlabel("t")
# plt.ylabel("x")


# Visualize dataset
fig, axs = plt.subplots(3)
fig.suptitle("Dynamic behavior of the dimensions of a CDV flow")

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

fig, axs = plt.subplots(3)
fig.suptitle("Dynamic behavior of the dimensions of a CDV flow")

plt.subplot(3,1,1)
plt.plot(t, x[:,3])
plt.ylabel("$x_4$")
plt.xlabel("t")
plt.subplot(3,1,2)
plt.plot(t, x[:,4])
plt.ylabel("$x_5$")
plt.xlabel("t")
plt.subplot(3,1,3)
plt.plot(t, x[:,5])
plt.ylabel("$x_6$")
plt.xlabel("t")

# phase space
plt.figure(figsize=(6,6))
plt.scatter(x[:,0], x[:,3])
plt.ylabel("$x_4$")
plt.xlabel("$x_1$")

# Phase space - 5 first (most energetic) POD modes
# Compute POD
# num_modes = 5
# POD_res = mr.compute_POD_arrays_snaps_method(
#     x.transpose(), list(mr.range(num_modes)))
# modes = POD_res.modes   #array whose columns are POD modes.
# #eigvals = POD_res.eigvals
# x_projected = numpy.matmul(x,modes)
# x = x_projected

# print(modes)
# print(eigvals)

# Tesselation
M = 20  # number of divisions for tesselation, uniform for all dimensions
tess_ind, extr_id = tesselate(x,M,0)    # where 0 indicates the dimension by which the extreme event should be identified - here doesn't matter because we don't have an extreeme event

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
# one run
while int(np.size(np.unique(np.array(list(P_community_old.values())))))>25 and int_id<15: # condition
    int_id=int_id+1
    P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old, P_old, D_nodes_in_clusters)
    # print(np.sum(D_nodes_in_clusters,axis=0).tolist())

# Visualize clustered graph
plt.figure()
nx.draw_kamada_kawai(P_graph_old,with_labels=True)

#NEW MODULE - time series with cluster affiliation
# translate datapoints to cluster number affiliation
tess_ind_cluster = np.zeros_like(tess_ind_trans)
for point in np.unique(tess_ind_trans):     # take all unique points in tesselated space
    cluster_aff = int(D_nodes_in_clusters.col[D_nodes_in_clusters.row==point])  # find affiliated cluster
    tess_ind_cluster[tess_ind_trans==point] = cluster_aff

fig, axs = plt.subplots(3)
fig.suptitle("Dynamic behavior of the dimensions of a CDV flow")

plt.subplot(3,1,1)
plt.plot(t, x[:,0])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_1$")
plt.xlabel("t")
plt.subplot(3,1,2)
plt.plot(t, x[:,1])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_2$")
plt.xlabel("t")
plt.subplot(3,1,3)
plt.plot(t, x[:,2])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_3$")
plt.xlabel("t")

fig, axs = plt.subplots(3)
fig.suptitle("Dynamic behavior of the dimensions of a CDV flow")

plt.subplot(3,1,1)
plt.plot(t, x[:,3])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_4$")
plt.xlabel("t")
plt.subplot(3,1,2)
plt.plot(t, x[:,4])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_5$")
plt.xlabel("t")
plt.subplot(3,1,3)
plt.plot(t, x[:,5])
for i in range(len(tess_ind_cluster)-1):
    if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
        plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
plt.ylabel("$x_6$")
plt.xlabel("t")

# phase space
plt.figure(figsize=(6,6))
for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
    plt.scatter(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,3])  # I should relate somehow s to N and the fig size
plt.ylabel("$x_4$")
plt.xlabel("$x_1$")

plt.show()