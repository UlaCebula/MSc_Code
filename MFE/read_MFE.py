# script for calcuations for the Moehlis-Faisst-Eckhardt equations - 9 dimensional
# Urszula Golyska 2022

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from MFE_func import *
import h5py
from my_func import *
import networkx as nx
from modularity_maximization import spectralopt
import scipy.sparse as sp
import seaborn as sns

plt.close('all') # close all open figures

# #####BURST#########
# type='burst'
# filename = 'MFE_Re400_T10000.h5'
# t,u = read_Fourier(filename)
#
# x = to_burst(u)
# dim = 3
# extr_dim = 2    # define burst as the extreme dimension
#
# # Tesselation
# M = 20
#
# MFE_process(t,x,dim,M,extr_dim,type)
# plt.show()

#####DISSIPATION#########
type='dissipation'
filename = 'MFE_Re600'
dt = 0.25
t,x = read_DI(filename, dt)
dim = 2
extr_dim = 0    # define dissipation as the extreme dimension

# Tesselation
M = 20

MFE_process(t,x,dim,M,extr_dim,type)
plt.show()