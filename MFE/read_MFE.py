import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import h5py
    

hf = h5py.File('MFE_Re600_T80000_CUT.h5','r')

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
    
plt.figure(1)
plt.subplot(511)
plt.plot(t,u[:,0])
plt.subplot(512)
plt.plot(t,u[:,1])
plt.subplot(513)
plt.plot(t,u[:,2])
plt.subplot(514)
plt.plot(t,u[:,3])
plt.subplot(515)
plt.plot(t,u[:,4])

plt.figure(2)
plt.subplot(511)
plt.plot(t,u[:,5])
plt.subplot(512)
plt.plot(t,u[:,6])
plt.subplot(513)
plt.plot(t,u[:,7])
plt.subplot(514)
plt.plot(t,u[:,8])

plt.show()


    
