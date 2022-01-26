# script for generating flow data for the Lorenz attractor model for Rayleigh-Bernard convection - 3 dimensional
# Urszula Golyska 2022

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from LA_tesselation import tesselate


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

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x[:,0], x[:,1], x[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

N = 10000  # number of divisions for tesselation, uniform for all dimensions
tess_ind = tesselate(x,N)    # output - for sparse approach -indices of occupied spaces, for non-sparse approach - matrix with occupied spaces
# A = coo_matrix(ind, np.ones(N), shape=(N * 3)) # create sparse matrix at given indices

#for visualization only - works only for the 3d case
# full coorindate arrays
ax = plt.axes(projection='3d')
ax.scatter3D(tess_ind[:,0], tess_ind[:,1], tess_ind[:,2])
plt.show()