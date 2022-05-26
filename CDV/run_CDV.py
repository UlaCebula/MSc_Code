# clustering and extreme event detection for the Charney-DeVore equations - 6 dimensional
# Urszula Golyska 2022
# data generation part by Anh Khoa Doan
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
    dxdt = np.zeros(x.shape)
    dxdt[:,0] = gamma_m_star[0]*x[:,2] - C*(x[:,0] - x1s)
    dxdt[:,1] = -(alpha[0]*x[:,0] - beta[0])*x[:,2] - C*x[:,1] - delta[0]*x[:,3]*x[:,5] ## WHY IS THERE A MINUS HERE?
    dxdt[:,2] = (alpha[0]*x[:,0] - beta[0])*x[:,1] - gamma_m[0]*x[:,0] - C*x[:,2] + delta[0]*x[:,3]*x[:,4]
    dxdt[:,3] = gamma_m_star[1]*x[:,5] - C*(x[:,3] - x4s) + epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
    dxdt[:,4] = -(alpha[1]*x[:,0] - beta[1])*x[:,5] - C*x[:,4] - delta[1]*x[:,2]*x[:,3]
    dxdt[:,5] = (alpha[1]*x[:,0] - beta[1])*x[:,4] - gamma_m[1]*x[:,3] - C*x[:,5] + delta[1]*x[:,3]*x[:,1]
    #dxdt = np.matmul(x,L) + NL(x) + b
    return dxdt

def save_data_h5(fln,x,t,dt,x1s,x4s,C,beta0,gamma,m,alpha,beta,delta,gamma_m,gamma_m_star,epsilon):
    hf = h5py.File(fln,'w')
    hf.create_dataset('x',data=x)
    hf.create_dataset('t',data=t)
    hf.create_dataset('dt',data=dt)
    #hf.create_dataset('L',data=L)
    hf.create_dataset('x1s',data=x1s)
    hf.create_dataset('x4s',data=x4s)
    hf.create_dataset('C',data=C)
    hf.create_dataset('beta0',data=beta0)
    hf.create_dataset('gamma',data=gamma)
    #hf.create_dataset('b',data=b)
    hf.create_dataset('m',data=m)
    hf.create_dataset('alpha',data=alpha)
    hf.create_dataset('beta',data=beta)
    hf.create_dataset('delta',data=delta)
    hf.create_dataset('gamma_m',data=gamma_m)
    hf.create_dataset('gamma_m_star',data=gamma_m_star)
    hf.create_dataset('epsilon',data=epsilon)
    hf.close()

plt.close('all') # close all open figures
type='CDV'

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
# save_data_h5(fln,x,t,dt,x1s,x4s,C,beta0,gamma,m,alpha,beta,delta,gamma_m,gamma_m_star,epsilon)

M = 20  # tesselation
extr_dim = []   # define no extreme dimension

plotting = True
min_clusters=30
max_it=10

extreme_event_identification_process(t,x,dim,M,extr_dim,type, min_clusters, max_it, 'classic',7,plotting, False)
plt.show()

#
# #NEW MODULE - time series with cluster affiliation
# # translate datapoints to cluster number affiliation
# tess_ind_cluster = np.zeros_like(tess_ind_trans)
# for point in np.unique(tess_ind_trans):     # take all unique points in tesselated space
#     cluster_aff = int(D_nodes_in_clusters.col[D_nodes_in_clusters.row==point])  # find affiliated cluster
#     tess_ind_cluster[tess_ind_trans==point] = cluster_aff
#
# fig, axs = plt.subplots(3)
# fig.suptitle("Dynamic behavior of the dimensions of a CDV flow")
#
# plt.subplot(3,1,1)
# plt.plot(t, x[:,0])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
# plt.ylabel("$x_1$")
# plt.xlabel("t")
# plt.subplot(3,1,2)
# plt.plot(t, x[:,1])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
# plt.ylabel("$x_2$")
# plt.xlabel("t")
# plt.subplot(3,1,3)
# plt.plot(t, x[:,2])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
# plt.ylabel("$x_3$")
# plt.xlabel("t")
#
# fig, axs = plt.subplots(3)
# fig.suptitle("Dynamic behavior of the dimensions of a CDV flow")
#
# plt.subplot(3,1,1)
# plt.plot(t, x[:,3])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
# plt.ylabel("$x_4$")
# plt.xlabel("t")
# plt.subplot(3,1,2)
# plt.plot(t, x[:,4])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
# plt.ylabel("$x_5$")
# plt.xlabel("t")
# plt.subplot(3,1,3)
# plt.plot(t, x[:,5])
# for i in range(len(tess_ind_cluster)-1):
#     if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
#         plt.axvline(x=(t[i]+t[i+1])/2, color = 'r', linestyle = '--')
# plt.ylabel("$x_6$")
# plt.xlabel("t")
#
# # phase space
# plt.figure(figsize=(6,6))
# for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
#     plt.scatter(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,3])  # I should relate somehow s to N and the fig size
# plt.ylabel("$x_4$")
# plt.xlabel("$x_1$")
#
# plt.show()