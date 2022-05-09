import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation

# plt.close('all') # close all open figures
def reconstruct_velocity(hf):
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
    # dx = xx[1]-xx[0]
    # dy = zz[1]-zz[0]

    N8 = 2 * np.sqrt(2) / np.sqrt((alpha**2 + gamma**2) * (4 * alpha**2 + 4 * gamma**2 + np.pi**2));
    uu = np.zeros((3, 9))

    u = np.zeros((Nx, Nz, Ny, 3, Nt)) #Nx*Ny*t for 2d
    k = 1
    # mov_name = 'MFE_Re600_breakdown.gif'

    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    # fig, ax = plt.subplots()

    # max_u = np.zeros_like(t)

    for it in range(Nt):#np.linspace(39500, 41000, 10, endpoint=True, dtype=int):
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

                u[ix, iz, iy,:, 0] = np.matmul(uu,a[it,:])

        # plt.contourf(xx, zz, np.squeeze(u[:,:, 0, 1, 0]))
        # # quiver(xx, zz, squeeze(u(:,:, 1, 1, 1)), squeeze(u(:,:, 1, 3, 1)))
        # # colorbar
        # text = 't = ' + str(it * 0.25)
        # plt.xlabel(text)
        # # M(k) = getframe(hf);
        # plt.show()
        # # im = frame2im(M(k));
        # # [imind, cm] = rgb2ind(im, 256);
        #
        # # if k == 1:
        # #     imwrite(imind, cm, mov_name, 'gif', 'Loopcount', inf);
        # # else:
        # #     imwrite(imind, cm, mov_name, 'gif', 'WriteMode', 'append');

        # max_u[it] = max(u[:, :, :,2, 0].flatten())
        k = k + 1
    # plt.plot(t,max_u)   # max of u in whole domain
    # plt.show()

    ## Calculate dissipation
    print("done")

    return u

def compute_grad():
    #two components of velocity - just 2d
    dudx = np.zeros((Nx,Nz,Nt))
    dudy = np.zeros((Nx,Nz,Nt))
    dvdx = np.zeros((Nx,Nz,Nt))
    dvdy = np.zeros((Nx,Nz,Nt))
    dwdx = np.zeros((Nx,Nz,Nt))
    dwdy = np.zeros((Nx,Nz,Nt))

    Diss = np.zeros_like(t)
    # FD discertization to find gradients
    for i in range(Nt):
        u_loc = np.squeeze(u[:, :, 0, 0, i])
        v_loc = np.squeeze(u[:, :, 0, 1, i])
        w_loc = np.squeeze(u[:, :, 0, 2, i])

        # for inner points
        # du/dx
        dudx[:,:,i] = (np.append(u_loc[1:, :],np.zeros((1,Nz)), axis=0)-np.append(np.zeros((1,Nz)),u_loc[:-1, :], axis=0))/(2*dx)  # shift the matrices for a quick FD approx
        # dv/dx
        dvdx[:,:,i] = (np.append(v_loc[1:, :],np.zeros((1,Nz)), axis=0)-np.append(np.zeros((1,Nz)),v_loc[:-1, :], axis=0))/(2*dx)  # shift the matrices
        # dw/dx
        dwdx[:, :, i] = (np.append(w_loc[1:, :], np.zeros((1, Nz)), axis=0) - np.append(np.zeros((1, Nz)), w_loc[:-1, :],axis=0)) / 2 * dx)  # shift the matrices

        # du/dy
        dudy[:,:,i] = (np.append(u_loc[:, 1:],np.zeros((Nz,1)), axis=1)-np.append(np.zeros((Nz,1)),u_loc[:, :-1], axis=1))/(2*dy)  # shift the matrices for a quick FD approx
        # dv/dy
        dvdy[:,:,i] = (np.append(v_loc[:, 1:],np.zeros((Nz,1)), axis=1)-np.append(np.zeros((Nz,1)),v_loc[:, :-1], axis=1))/(2*dy)  # shift the matrices for a quick FD approx
        # dw/dy
        dwdy[:,:,i] = (np.append(w_loc[:, 1:],np.zeros((Nz,1)), axis=1)-np.append(np.zeros((Nz,1)),w_loc[:, :-1], axis=1))/(2*dy)  # shift the matrices for a quick FD approx

        #boundary values
        dudx[]=
        dvdx[]=
        dwdx[]=
        dudy[]=
        dvdy[]=
        dwdy[]=

    return

def compute_diss():
    #next
    s_xx = 1/2*(dudx[:,:,i]+dudx[:,:,i])
    s_xy = 1/2*(dudy[:,:,i]+dvdx[:,:,i])
    s_yx = 1/2*(dvdx[:,:,i]+dudy[:,:,i])
    s_yy = 1/2*(dvdy[:,:,i]+dvdy[:,:,i])
    # dissipation
    nu = 1. #doesn't matter because it's a scalar constant the same for all time instants
    Diss[i] = nu * np.sum(dudx[:,:,i]*dudx[:,:,i])+np.sum(dudy[:,:,i]*dvdx[:,:,i])+np.sum(dvdy[:,:,i]*dvdy[:,:,i])+np.sum(dvdx[:,:,i]*dudy[:,:,i])

    return Diss


# print(np.nonzero(dudx.flatten()))
# print(np.nonzero(dudy.flatten()))
# print(np.nonzero(dvdx.flatten()))
# print(np.nonzero(dvdy.flatten()))
# #s
# plt.plot(t[:Nt],Diss[:Nt])
# plt.show()
# fln = 'MFE_Re600_DATA_dissipation.h5'
# f = h5py.File(fln,'w')
# f.create_dataset('D',data=Diss)
# f.close()


###################MAIN################

file_name = h5py.File('MFE_Re600_DATA.h5','r')
u = reconstruct_velocity(file_name)
np.save('MFE_Re600_velocity.npy', u)