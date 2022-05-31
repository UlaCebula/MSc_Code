# script for generating flow data for the Lorenz attractor model for Rayleigh-Bernard convection - 3 dimensional
# Urszula Golyska 2022

from my_func import *

def LA_dt(x): # explicit Euler scheme
    dxdt = np.zeros(x.shape)
    dxdt[0] = sigma*(x[1] - x[0])
    dxdt[1] = x[0]*(r-x[2])-x[1]
    dxdt[2] = x[0]*x[1]-b*x[2]
    return dxdt

plt.close('all') # close all open figures
type='LA'
# model coefficients - same as in Kaiser et al. 2014
sigma = 10
b = 8/3
r = 28

# time discretization
t0 = 0.0
tf = 110.0
dt = 0.01
t = np.arange(t0,tf,dt)
N = np.size(t)

dim = 3   # three dimensional
x = np.zeros((N,dim))
x[0,:] = np.array([0, 1.0, 1.05]) # initial condition

for i in range(N-1):
    q = LA_dt(x[i,:])
    x[i+1,:] = x[i,:] + dt*q

# delete first 500 ts to avoid numerical instabilities
x = x[500:,:]
t = t[500:]

M=20
extr_dim = []   # dimension of extreme event, here none

clusters, D, P = extreme_event_identification_process(t,x,dim,M,extr_dim,type, 20, 20, 'classic',7,True, False)
plt.show()