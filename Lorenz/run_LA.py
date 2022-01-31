# script for generating flow data for the Lorenz attractor model for Rayleigh-Bernard convection - 3 dimensional
# Urszula Golyska 2022

import numpy as np
import matplotlib.pyplot as plt
from my_func import tesselate, trans_matrix


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
tf = 100.0
dt = 0.01
t = np.arange(t0,tf,dt)
N = np.size(t)

x = np.zeros((N,3))
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
plt.show()


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