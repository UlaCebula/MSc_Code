# script for reconstructing velocity field, calculating dissipation and kinetic energy from MFE data
# Urszula Golyska 2022

import numpy as np
import matplotlib.pyplot as plt
import h5py

def reconstruct_velocity(Nx, Ny, Nz, xx, yy, zz, a, alpha, gamma):
    N8 = 2 * np.sqrt(2) / np.sqrt((alpha**2 + gamma**2) * (4 * alpha**2 + 4 * gamma**2 + np.pi**2));
    uu = np.zeros((3, 9))

    u = np.zeros((Nx, Nz, Ny, 3, Nt)) #Nx*Ny*t for 2d

    for it in range(Nt):    #np.linspace(39500, 41000, 10, endpoint=True, dtype=int):
        iy = 0
        for ix in range(len(xx)):
            for iz in range(len(zz)):
                x = xx[ix]
                z = zz[iz]
                y = yy[iy]
                uu[:, 0] = [np.sqrt(2) * np.sin(np.pi * y / 2), 0, 0]
                uu[:, 1] = [4 / np.sqrt(3) * (np.cos(np.pi * y / 2))**2 * np.cos(gamma * z), 0, 0] # the square has to be element-wise
                uu[:, 2] = [(2 / np.sqrt(4 * gamma**2 + np.pi**2)) * temp for temp in [0, 2 * gamma * np.cos(np.pi * y / 2) * np.cos(np.pi * z / 2), np.pi * np.sin(np.pi * y / 2) * np.sin(gamma * z)]]
                uu[:, 3] = [0, 0, 4 / np.sqrt(3) * np.cos(alpha * x) * (np.cos(np.pi * y / 2))**2]
                uu[:, 4] = [0, 0, 2 * np.sin(alpha * x) * np.sin(np.pi * y / 2)]
                uu[:, 5]  = [((4 * np.sqrt(2)) / np.sqrt(3 * (alpha**2 + gamma**2)))*temp for temp in [ -gamma * np.cos(alpha * x) * (np.cos(np.pi * y / 2))**2 * np.sin(gamma * z), 0, alpha * np.sin(alpha * x) * (np.cos(np.pi * y / 2))**2 * np.cos(gamma * z)]]
                uu[:, 6]  = [((2 * np.sqrt(2)) / np.sqrt(alpha**2 + gamma**2))*temp for temp in [gamma * np.sin(alpha * x) * np.sin(np.pi * y / 2) * np.sin(gamma * z), 0, alpha * np.cos(alpha * x) * np.sin(np.pi * y / 2) * np.cos(gamma * z)]]
                uu[:, 7]  = [N8*temp for temp in [np.pi * alpha * np.sin(alpha * x) * np.sin(np.pi * y / 2) * np.sin(gamma * z), 2 * (alpha**2 + gamma**2) * np.cos(alpha * x) * np.cos(np.pi * y / 2) * np.sin(gamma * z), -np.pi * gamma * np.cos(alpha * x) * np.sin(np.pi * y / 2) * np.cos(gamma * z)]]
                uu[:, 8]  = [np.sqrt(2) * np.sin(3 * np.pi * y / 2), 0, 0]

                u[ix, iz, iy,:, it] = np.matmul(uu,a[it,:])
    return u

def compute_grad(Nx,Nz,Nt,dx,dy,u):     # change to have the possibility of having Ny>1
    #two components of velocity - just 2d
    dudx = np.zeros((Nx,Nz,Nt))
    dudy = np.zeros((Nx,Nz,Nt))
    dvdx = np.zeros((Nx,Nz,Nt))
    dvdy = np.zeros((Nx,Nz,Nt))
    dwdx = np.zeros((Nx,Nz,Nt))
    dwdy = np.zeros((Nx,Nz,Nt))

    # FD discertization to find gradients
    for i in range(Nt):
        u_loc = np.squeeze(u[:, :, 0, i])   #u is squeezed already
        v_loc = np.squeeze(u[:, :, 1, i])
        w_loc = np.squeeze(u[:, :, 2, i])

        # for inner points
        # du/dx
        dudx[:,:,i] = (np.append(u_loc[1:, :],np.zeros((1,Nz)), axis=0)-np.append(np.zeros((1,Nz)),u_loc[:-1, :], axis=0))/(2*dx)  # shift the matrices for a quick FD approx
        # dv/dx
        dvdx[:,:,i] = (np.append(v_loc[1:, :],np.zeros((1,Nz)), axis=0)-np.append(np.zeros((1,Nz)),v_loc[:-1, :], axis=0))/(2*dx)  # shift the matrices
        # dw/dx
        dwdx[:, :, i] = (np.append(w_loc[1:, :], np.zeros((1, Nz)), axis=0) - np.append(np.zeros((1, Nz)), w_loc[:-1, :],axis=0)) / (2 * dx)  # shift the matrices

        # du/dy
        dudy[:,:,i] = (np.append(u_loc[:, 1:],np.zeros((Nz,1)), axis=1)-np.append(np.zeros((Nz,1)),u_loc[:, :-1], axis=1))/(2*dy)  # shift the matrices for a quick FD approx
        # dv/dy
        dvdy[:,:,i] = (np.append(v_loc[:, 1:],np.zeros((Nz,1)), axis=1)-np.append(np.zeros((Nz,1)),v_loc[:, :-1], axis=1))/(2*dy)  # shift the matrices for a quick FD approx
        # dw/dy
        dwdy[:,:,i] = (np.append(w_loc[:, 1:],np.zeros((Nz,1)), axis=1)-np.append(np.zeros((Nz,1)),w_loc[:, :-1], axis=1))/(2*dy)  # shift the matrices for a quick FD approx

        ## boundary values- 1st order
        dudx[0,:,i]= (u_loc[1,:]-u_loc[0,:])/dx
        dudx[-1,:,i]= (u_loc[-1,:]-u_loc[-2,:])/dx
        dvdx[0,:,i]= (v_loc[1,:]-v_loc[0,:])/dx
        dvdx[-1,:,i]= (v_loc[-1,:]-v_loc[-2,:])/dx
        dwdx[0,:,i]= (w_loc[1,:]-w_loc[0,:])/dx
        dwdx[-1,:,i]= (w_loc[-1,:]-w_loc[-2,:])/dx

        dudy[:,0,i]= (u_loc[:,1]-u_loc[:,0])/dy
        dudy[:,-1,i]= (u_loc[:,-1]-u_loc[:,-2])/dy
        dvdy[:,0,i]= (v_loc[:,1]-v_loc[:,0])/dy
        dvdy[:,-1,i]= (v_loc[:,-1]-v_loc[:,-2])/dy
        dwdy[:,0,i]= (w_loc[:,1]-w_loc[:,0])/dy
        dwdy[:,-1,i]= (w_loc[:,-1]-w_loc[:,-2])/dy

    return dudx, dudy, dvdx, dvdy, dwdx, dwdy

def compute_diss(Nx, Nz, Nt, dx,dy,u):
    Diss = np.zeros((Nt,1))
    nu = 1.  # doesn't matter because it's a scalar constant the same for all time instants

    # calculate gradients
    dudx, dudy, dvdx, dvdy, dwdx, dwdy = compute_grad(Nx, Nz, Nt, dx,dy, u)

    # dissipation
    for i in range(Nt):
        Diss[i] = nu * np.sum(dudx[:,:,i]*dudx[:,:,i])+np.sum(dudy[:,:,i]*dvdx[:,:,i])+np.sum(dvdy[:,:,i]*dvdy[:,:,i])+np.sum(dvdx[:,:,i]*dudy[:,:,i])

    return Diss

###################MAIN################
plt.close('all') # close all open figures

hf = h5py.File('MFE_Re600_DATA.h5','r')
Lx = np.array(hf.get('/Lx'))
Lz = np.array(hf.get('/Lz'))

Lx = np.array(hf.get('/Lx'))
Lz = np.array(hf.get('/Lz'))
Re = np.array(hf.get('/Re'))
alpha = np.array(hf.get('/alpha'))
beta = np.array(hf.get('/beta'))
dt = np.array(hf.get('/dt'))
gamma = np.array(hf.get('/gamma'))
a = np.array(hf.get('/u'))
t = np.array(hf.get('/t'))

Nt = np.size(t)

Nx = 20
Nz = 20
Ny = 1

xx = np.linspace(0, Lx, Nx, endpoint=True)
zz = np.linspace(0, Lz, Nz, endpoint=True)
yy = [0]
dx = xx[1]-xx[0]
dy = zz[1]-zz[0]

# To reconstruct velocity
u = reconstruct_velocity(Nx, Ny, Nz, xx, yy, zz, a, alpha, gamma)
np.save('MFE_Re600_velocity.npy', u)

# Read and check velocity
# u = np.load('MFE_Re600_velocity.npy')
# t = np.arange(len(u[0,0,0,:]))
u = np.squeeze(u[:,:,0,:,:]) #because we have Ny=1

# u component in the middle of the domain
plt.figure()
plt.plot(t, u[9,9,0,:])   #u velocity components at random middle point (9,9)
plt.show()


# Calculate dissipation
# u = np.load('MFE_Re600_velocity.npy')
# Nt = np.size(u[0,0,0,:])
D = compute_diss(Nx, Nz, Nt, dx,dy, u)
np.save('MFE_Re600_dissipation.npy', D)

# Read and check dissipation
# D = np.load('MFE_Re600_dissipation.npy')
# t = np.arange(len(D))

plt.figure()
plt.plot(t, D)
plt.show()

# Calculate energy
# u = np.load('MFE_Re600_velocity.npy')
# Nt = np.size(u[0,0,0,:])
E = np.zeros((Nt,1))

for i in range(Nt):
    E[i] = np.sum(u[:,:,0,i]**2)+np.sum(u[:,:,1,i]**2)+np.sum(u[:,:,2,i]**2)
np.save('MFE_Re600_energy.npy', E)

plt.figure()
plt.plot(t, E)

plt.figure()
plt.plot(E,D)
plt.show()

