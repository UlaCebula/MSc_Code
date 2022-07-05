# read and visualize data for Kolmogorov flow

import numpy as np
import matplotlib.pyplot as plt
import h5py
from my_func import *

class Kol2D_odd(object):
    """
    N: resolution of grid used; number of grids (single direction) = (2N+1)
    Re: Reynolds number
    n: wavenumber of external forcing in x direction
    wave numbers are arranged such that 0 is in the center
    """

    def __init__(self, Re=40, n=4, N=6):

        self.N = N
        self.grid_setup(N)
        self.grids = 2 * N + 1
        self.Re = Re
        self.fx = np.fft.fftshift(np.fft.fft2(np.sin(n * self.yy)))

        # aa = np.fft.ifft2(np.fft.ifftshift(self.fx))
        # print(aa.real)
        # print(aa.imag)

    def grid_setup(self, N):

        # physical grid
        x = np.linspace(0, 2 * np.pi, 2 * N + 2)
        x = x[:-1]
        self.xx, self.yy = np.meshgrid(x, x)

        # wavenumbers
        k = np.arange(-N, N + 1)
        self.kk1, self.kk2 = np.meshgrid(k, k)
        self.kk = self.kk1 ** 2 + self.kk2 ** 2

        # parameters for divergence-free projection (Fourier domain)
        self.p1 = self.kk2 ** 2 / self.kk
        self.p2 = -self.kk1 * self.kk2 / self.kk
        self.p3 = self.kk1 ** 2 / self.kk

        # differentiation (Fourier domain)
        self.ddx = 1j * self.kk1
        self.ddy = 1j * self.kk2

        # matrix for converting u,v to a and vice versa: u = a*pu, v = a*pv
        self.pu = self.kk2 / np.sqrt(self.kk)
        self.pu[self.N, self.N] = 0
        self.pv = -self.kk1 / np.sqrt(self.kk)
        self.pv[self.N, self.N] = 0

    def proj_DF(self, fx_h, fy_h):  # divergence free projection

        ux_h = self.p1 * fx_h + self.p2 * fy_h
        uy_h = self.p2 * fx_h + self.p3 * fy_h

        # boundary conditions
        if fx_h.ndim == 2:
            ux_h[self.N, self.N] = 0
            uy_h[self.N, self.N] = 0

        elif fx_h.ndim == 3:
            ux_h[:, self.N, self.N] = 0
            uy_h[:, self.N, self.N] = 0

        return ux_h, uy_h

    def uv2a(self, u_h, v_h):  # unified Fourier coefficients a(x,t)

        a_h = u_h / self.pu
        a_v = v_h / self.pv

        if u_h.ndim == 2:
            a_h[self.N] = a_v[self.N]
            a_h[self.N, self.N] = 0
        elif u_h.ndim == 3:
            a_h[:, self.N, :] = a_v[:, self.N, :]
            a_h[:, self.N, self.N] = 0

        return a_h

    def a2uv(self, a_h):

        return a_h * self.pu, a_h * self.pv

    def vort(self, u_h, v_h):  # calculate vorticity

        return self.ddy * u_h - self.ddx * v_h

    def dissip(self, u_h, v_h):  # calculate dissipation

        w_h = self.vort(u_h, v_h)
        D = np.sum(w_h * w_h.conjugate(), axis=(-1, -2))
        D = np.squeeze(D) / self.Re / self.grids ** 4

        return D.real

    def dynamics(self, u_h, v_h):

        fx_h = -self.ddx * self.aap(u_h, u_h) - self.ddy * self.aap(u_h, v_h) + self.fx
        fy_h = -self.ddx * self.aap(u_h, v_h) - self.ddy * self.aap(v_h, v_h)

        Pfx_h, Pfy_h = self.proj_DF(fx_h, fy_h)

        du_h = -self.kk * u_h / self.Re + Pfx_h
        dv_h = -self.kk * v_h / self.Re + Pfy_h

        return du_h, dv_h

    def dynamics_a(self, a_h):

        u_h, v_h = self.a2uv(a_h)
        du_h, dv_h = self.dynamics(u_h, v_h)
        da_h = self.uv2a(du_h, dv_h)

        return da_h

    def random_field(self, A_std, A_mag, c1=0, c2=3):

        '''
            generate a random field whose energy is normally distributed
            in Fourier domain centered at wavenumber (c1,c2) with random phase
        '''

        A = A_mag * 4 * self.grids ** 2 * np.exp(-(self.kk1 - c1) ** 2 -
                                                 (self.kk2 - c2) ** 2 / 2 / A_std ** 2) / np.sqrt(
            2 * np.pi * A_std ** 2)
        u_h = A * np.exp(1j * 2 * np.pi * np.random.rand(self.grids, self.grids))
        v_h = A * np.exp(1j * 2 * np.pi * np.random.rand(self.grids, self.grids))

        u = np.fft.irfft2(np.fft.ifftshift(u_h), s=u_h.shape[-2:])
        v = np.fft.irfft2(np.fft.ifftshift(v_h), s=v_h.shape[-2:])

        u_h = np.fft.fftshift(np.fft.fft2(u))
        v_h = np.fft.fftshift(np.fft.fft2(v))

        u_h, v_h = self.proj_DF(u_h, v_h)

        return u_h, v_h

    def plot_vorticity(self, u_h, v_h, wmax=None, subplot=False):

        w_h = self.vort(u_h, v_h)
        w = np.fft.ifft2(np.fft.ifftshift(w_h))
        w = w.real

        # calculate color axis limit if not specified
        if not wmax:
            wmax = np.ceil(np.abs(w).max())
        wmin = -wmax

        ## plot with image
        tick_loc = np.array([0, .5, 1, 1.5, 2]) * np.pi
        tick_label = ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        im = plt.imshow(w, cmap='RdBu', vmin=wmin, vmax=wmax,
                        extent=[0, 2 * np.pi, 0, 2 * np.pi],
                        interpolation='spline36', origin='lower')
        plt.xticks(tick_loc, tick_label)
        plt.yticks(tick_loc, tick_label)
        if subplot:
            plt.colorbar(im, fraction=.046, pad=.04)
            plt.tight_layout()
        else:
            plt.colorbar()

    def plot_quiver(self, u_h, v_h):

        u = np.fft.ifft2(np.fft.ifftshift(u_h)).real
        v = np.fft.ifft2(np.fft.ifftshift(v_h)).real

        Q = plt.quiver(self.xx, self.yy, u, v, units='width')

        tick_loc = np.array([0, .5, 1, 1.5, 2]) * np.pi
        tick_label = ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']

        plt.xticks(tick_loc, tick_label)
        plt.yticks(tick_loc, tick_label)

    def aap(self, f1, f2):  # anti-aliased product

        ndim = f1.ndim
        assert ndim < 4, 'input dimensions is greater than 3.'
        if ndim == 2:
            f1_h, f2_h = np.expand_dims(f1, axis=0).copy(), np.expand_dims(f2, axis=0).copy()
        elif ndim == 3:
            f1_h, f2_h = f1.copy(), f2.copy()

        sz2 = 4 * self.N + 1
        ff1_h = np.zeros((f1_h.shape[0], sz2, sz2), dtype=np.complex128)
        ff2_h = np.zeros((f1_h.shape[0], sz2, sz2), dtype=np.complex128)

        idx1, idx2 = self.N, 3 * self.N + 1
        ff1_h[:, idx1:idx2, idx1:idx2] = f1_h
        ff2_h[:, idx1:idx2, idx1:idx2] = f2_h

        ff1 = np.fft.irfft2(np.fft.ifftshift(ff1_h), s=ff1_h.shape[-2:])
        ff2 = np.fft.irfft2(np.fft.ifftshift(ff2_h), s=ff1_h.shape[-2:])  # must take real part or use irfft2

        pp_h = (sz2 / self.grids) ** 2 * np.fft.fft2(ff1 * ff2)
        pp_h = np.fft.fftshift(pp_h)

        p_h = pp_h[:, idx1:idx2, idx1:idx2]

        if ndim == 2:
            p_h = p_h[0, :, :]

        return p_h

plt.close('all')  # close all open figures

############### Read data ######################
Re = 40.    # Reynolds number Re=30 has extreme events
N = 8 # modes (pair)
n = 4   #what is this???
T = 5000
dt =.01
# t1 = np.arange(0,T,dt)

fln = 'Kolmogorov_Re' + str(Re) + '_T' + str(T) + '_DT01_fourier_ready.h5'
hf = h5py.File(fln, 'r')
t = np.array(hf.get('t'))
# N2 = np.array(hf.get('/N'))
# kx = np.array(hf.get('/kx'))
# x = np.array(hf.get('/x'))
# xx = np.array(hf.get('/xx'))
# yy = np.array(hf.get('/yy'))
# vort = np.array(hf.get('/vort'))
# u = np.array(hf.get('/u'))
# v = np.array(hf.get('/v'))
Diss = np.array(hf.get('/Diss'))
# a_10 = np.array(hf.get('/four_a10'))
I = np.array(hf.get('/I'))
four_uu = np.array(hf.get('/four_uu_real'))

hf.close()

# D = np.array(D)
# plt.figure()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)
# plt.plot(t,D)
# plt.xlabel("t")
# plt.ylabel("D")
# plt.xlim(0,T)

########## getting the parameter that we want () ############
# is this the Fourier mode? I think so
# plt.figure()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)
# plt.plot(I,D)
# plt.xlabel("I")
# plt.ylabel("D")

# plt.figure()
# plt.plot(t,I)
#
# plt.figure()
# plt.plot(t,Diss)

# plt.show()

# plot the quantity of interest - absolute value of Fourier mode a(1,0)
# plt.figure()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)
# plt.plot(t,a_10)
# plt.xlabel("t")
# plt.ylabel("$|a(1,0)|$")
# plt.xlim(0,T)

type='kolmogorov'
dim = 4
Diss = Diss.reshape((len(Diss),1))
I = I.reshape((len(Diss),1))
x = np.append(Diss, I, axis=1)      # same as in Farazmand and sapsis - this plus triad with k_f = 4, where this is the mean flow
# x = np.append(x,four_uu[1,0,:].reshape(len(t),1), axis=1)
x = np.append(x,four_uu[0,4,:].reshape(len(t),1), axis=1)
x = np.append(x,four_uu[1,4,:].reshape(len(t),1), axis=1) # this is the faulty one
# for i in range(0,9):
#     for j in range(0,9):
#         x = np.append(x,four_uu[i,j,:].reshape(len(t),1), axis=1)
extr_dim = np.arange(0,dim) #np.arange(0,83)    # define both dissipation and energy as the extreme dimensions

####TEMPORARY### Plot all extreme dimensions on one plot normalised
# colors = ['r', 'g', 'b', 'y', 'c', 'k']
# lines = ['-', '-', '--', ':', '-.']
# plt.figure()
# for i in range(5):
#     avg = np.mean(x[:,i])
#     maxi = np.max(x[:,i])
#     mini = np.min(x[:,i])
#     x[:,i] = np.divide((x[:,i]-avg),abs(maxi-mini))
#     plt.plot(t, x[:,i], color = colors[i], linestyle = lines[i])
# plt.legend(['D','k', 'a(1,0)', 'a(0,4)', 'a(1,4)'])
# plt.show()

# Tesselation
M = 20

plotting = True
min_clusters=30 #20
max_it=10

clusters, D, P = extreme_event_identification_process(t,x,dim,M,extr_dim,type, min_clusters, max_it, 'classic', 5,plotting, False)
# plt.show()

extr_clusters = np.empty(0, int)
for i in range(len(clusters)):  #print average times spend in extreme clusters
    loc_cluster = clusters[i]
    if loc_cluster.is_extreme==2:
        extr_clusters = np.append(extr_clusters, i)
    # print("Average time in cluster ", loc_cluster.nr, " is: ", loc_cluster.avg_time, " s")

# find all paths to extreme events
paths = find_extr_paths(extr_clusters,P)

min_prob = np.zeros((len(clusters)))
min_time = np.zeros((len(clusters)))
length = np.zeros((len(clusters)))

for i in range(len(clusters)):  # for each cluster
    # prob to extreme
    loc_prob,loc_time,loc_length = prob_to_extreme(i, paths, t[-1], P, clusters)
    min_prob[i] = loc_prob
    min_time[i] = loc_time
    length[i] = loc_length

plot_cluster_statistics(clusters, min_prob, min_time, length)
plt.show()

# # input other (small) data and analyze it
# T = 5000
# fln = 'Kolmogorov_Re' + str(Re) + '_T' + str(T) + '_DT01.h5'
# hf = h5py.File(fln, 'r')
# t_new = np.array(hf.get('t'))
# Diss_new = np.array(hf.get('/Dissip'))
# I_new = np.array(hf.get('/E'))
# hf.close()
# Diss_new = Diss_new.reshape((len(Diss_new),1))
# I_new = I_new.reshape((len(I_new),1))
# x_new = np.append(Diss_new, I_new, axis=1)
#
# #tesselate the new data
# x_new_tess,temp= tesselate(x_new,M,[],5)    #tesselate function without extreme event id
# x_new_tess = tess_to_lexi(x_new_tess, M, dim)
#
# # cluster affiliation
# x_new_clusters = data_to_clusters(x_new_tess, D, x_new, clusters)
#
# # show time series with extreme events (real-time)
# fig, axs = plt.subplots(2)
# fig.suptitle("Real-time predictions")
#
# axs[0].set_xlim([t_new[0], t_new[-1]])
# axs[0].set_xlabel("t")
# axs[0].set_ylabel("D")
#
# # axs[1].set_xlim([t_new[0], t_new[-1]])
# axs[1].set_xlabel("t")
# axs[1].set_ylabel("extreme")
# axs[1].set_ylim([-0.5, 2.5])
#
# n_skip=10
# spacing = np.arange(0, len(t_new), n_skip, dtype=int)
# for i in range(len(spacing)):
#     if i!=0:
#         loc_clust = data_to_clusters(x_new_tess[spacing[i]], D, x, clusters)
#         axs[0].plot([t_new[spacing[i - 1]], t_new[spacing[i]]], [x_new[spacing[i - 1], 0], x_new[spacing[1], 0]], color='blue')
#
#         temp2=clusters[x_new_clusters[spacing[i]]].is_extreme
#         temp = clusters[x_new_clusters[spacing[i-1]]].is_extreme
#         # probability of transitioning to extreme event (shortest path)    # minimum time to extreme event
#         if temp2==2:
#             axs[1].plot([t_new[spacing[i-1]], t_new[spacing[i]]], [temp, temp2],color='red')
#
#         elif temp2==1:
#             axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
#                         [temp,
#                          temp2], color='orange')
#         else:
#             if temp==2:  # if previous was extreme
#                 axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
#                             [temp, temp2], color='red')
#             elif temp==1:   # if previous was precursor
#                 axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
#                             [temp, temp2], color='orange')
#             else:
#                 axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
#                         [temp, temp2], color='green')
#
#         # text = fig.text(0.05, 0.03, str(clusters[x_new_clusters[spacing[i]]].prob_to_extreme))
#         text = fig.text(0.05, 0.01, 'Probability: ' + str(min_prob[clusters[x_new_clusters[spacing[i]]].nr]))
#         text2 = fig.text(0.05, 0.05, 'Time: ' + str(min_time[clusters[x_new_clusters[spacing[i]]].nr]))
#         # axs[1].text(0,0,)
#         axs[1].set_xlim([t_new[spacing[i-1]]-n_skip*10, t_new[spacing[i]]+ n_skip*10])
#         plt.pause(0.001)
#         text.remove()
#         text2.remove()
# plt.show()
