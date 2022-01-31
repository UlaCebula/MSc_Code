# testing the created tesselation and transition probability matrix functions on a simple circle case
import numpy as np
import matplotlib.pyplot as plt
import math
from my_func import tesselate
from my_func import trans_matrix

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

# plt.figure(figsize=(13, 2))
# plt.plot(t,u1, '--')
# plt.plot(t,u2, '-.')
# plt.legend(['$u_1$','$u_2$'], loc="upper left")
# plt.grid('minor', 'both')
# plt.minorticks_on()
# plt.xlabel("t")
# plt.ylabel("u")
#
# plt.figure(figsize=(7, 7))
# plt.plot(u1,u2)
# plt.grid('minor', 'both')
# plt.minorticks_on()
# plt.xlabel("$u_1$")
# plt.ylabel("$u_2$")
#
# plt.show()

# combine into one matrix
u = np.hstack([np.reshape(u1, (len(t),1)),np.reshape(u2, (len(t),1))])

# tesselate
N = 20
tess_ind = tesselate(u,N)

# plt.figure(figsize=(7, 7))
# plt.scatter(tess_ind[:,0], tess_ind[:,1], s=100, marker='s', facecolors = 'None', edgecolor = 'blue')
# plt.grid('minor', 'both')
# plt.minorticks_on()
# plt.xlabel("$u_1$")
# plt.ylabel("$u_2$")
#
# plt.show()

# transition probability matrix
P = trans_matrix(tess_ind)  # create sparse transition probability matrix

# translate the 4d matrix into big 2d matrix
P_2d = P
# for i in P_2d:
#     if
# print(P)