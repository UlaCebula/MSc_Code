# script for calcuations for the Moehlis-Faisst-Eckhardt equations - 9 dimensional
# Urszula Golyska 2022
import h5py
import matplotlib.pyplot as plt

from my_func import *
import numpy.linalg as linalg
import time
import numpy as np

def MFE_get_param(alpha, beta, gamma, Re):
    """ Function for calculating the parameters of the MFE system

    :param alpha:
    :param beta:
    :param gamma:
    :param Re: Reynolds number of the system
    :return: returns 10 parameters, zeta and xi1-xi9
    """
    kag = np.sqrt(alpha ** 2. + gamma ** 2.)  # k alpha gamma
    kbg = np.sqrt(beta ** 2. + gamma ** 2.)  # k beta gamma
    kabg = np.sqrt(alpha ** 2. + beta ** 2. + gamma ** 2.)  # k alpha beta gamma
    k1 = (alpha * beta * gamma) / (kag * kbg)
    k2 = (alpha * beta * gamma) / (kag * kabg)

    # linear and forcing term
    zeta = np.array([beta ** 2., 4. * beta ** 2. / 3 + gamma ** 2., beta ** 2. + gamma ** 2.,
                     (3. * alpha ** 2. + 4 * beta ** 2.) / 3., alpha ** 2. + beta ** 2.,
                     (3. * alpha ** 2. + 4. * beta ** 2. + 3. * gamma ** 2.) / 3.,
                     alpha ** 2. + beta ** 2. + gamma ** 2., alpha ** 2. + beta ** 2. + gamma ** 2., 9. * beta ** 2.]
                    ) / Re
    zeta = np.diag(zeta)

    # non-linear coupling coefficients
    xi1 = np.array([np.sqrt(3. / 2.) * beta * gamma / kabg, np.sqrt(3. / 2.) * beta * gamma / kbg])

    xi2 = np.array([(5. * np.sqrt(2.) * gamma ** 2.) / (3. * np.sqrt(3) * kag), gamma ** 2. / (np.sqrt(6.) * kag),
                    k2 / np.sqrt(6.), xi1[1], xi1[1]])

    xi3 = np.array([2. * k1 / np.sqrt(6.),
                    (beta ** 2. * (3. * alpha ** 2. + gamma ** 2.) - 3. * gamma ** 2. * (alpha ** 2. + gamma ** 2.)) / (
                                np.sqrt(6.) * kag * kbg * kabg)])

    xi4 = np.array([alpha / np.sqrt(6.), 10. * alpha ** 2. / (3. * np.sqrt(6.) * kag), np.sqrt(3. / 2.) * k1,
                    np.sqrt(3. / 2.) * alpha ** 2. * beta ** 2. / (kag * kbg * kabg),
                    alpha / np.sqrt(6.)])

    xi5 = np.array([xi4[0], alpha ** 2. / (np.sqrt(6.) * kag), xi2[2], xi4[0], xi3[0]])

    xi6 = np.array(
        [xi4[0], xi1[0], (10. * (alpha ** 2. - gamma ** 2.)) / (3. * np.sqrt(6.) * kag), 2. * np.sqrt(2. / 3.) * k1,
         xi4[0], xi1[0]])

    xi7 = np.array([xi4[0], (gamma ** 2. - alpha ** 2.) / (np.sqrt(6.) * kag), k1 / np.sqrt(6.)])

    xi8 = np.array([2. * k2 / np.sqrt(6.), gamma ** 2. * (3. * alpha ** 2. - beta ** 2. + 3. * gamma ** 2.) / (
                np.sqrt(6.) * kag * kbg * kabg)])

    xi9 = np.array([xi1[1], xi1[0]])

    return zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9

def MFE_RHS(u, zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9):
    """Function for calculating the right hand side of the MFE system

    :param u: matrix of Fourier coefficients (of size Nt*9)
    :param zeta: system parameters
    :param xi1-xi9: full system coefficients
    :return: returns right hand side of the MFE system
    """
    RHS = - np.matmul(u, zeta)

    RHS[0] = RHS[0] + zeta[0, 0] - xi1[0] * u[5] * u[7] + xi1[1] * u[1] * u[2]

    RHS[1] = RHS[1] + xi2[0] * u[3] * u[5] - xi2[1] * u[4] * u[6] - xi2[2] * u[4] * u[7] - xi2[3] * u[0] * u[2] - xi2[
        4] * u[2] * u[8]
    RHS[2] = RHS[2] + xi3[0] * (u[3] * u[6] + u[4] * u[5]) + xi3[1] * u[3] * u[7]
    RHS[3] = RHS[3] - xi4[0] * u[0] * u[4] - xi4[1] * u[1] * u[5] - xi4[2] * u[2] * u[6] - xi4[3] * u[2] * u[7] - xi4[
        4] * u[4] * u[8]
    RHS[4] = RHS[4] + xi5[0] * u[0] * u[3] + xi5[1] * u[1] * u[6] - xi5[2] * u[1] * u[7] + xi5[3] * u[3] * u[8] + xi5[
        4] * u[2] * u[5]
    RHS[5] = RHS[5] + xi6[0] * u[0] * u[6] + xi6[1] * u[0] * u[7] + xi6[2] * u[1] * u[3] - xi6[3] * u[2] * u[4] + xi6[
        4] * u[6] * u[8] + xi6[5] * u[7] * u[8]
    RHS[6] = RHS[6] - xi7[0] * (u[0] * u[5] + u[5] * u[8]) + xi7[1] * u[1] * u[4] + xi7[2] * u[2] * u[3]
    RHS[7] = RHS[7] + xi8[0] * u[1] * u[4] + xi8[1] * u[2] * u[3]
    RHS[8] = RHS[8] + xi9[0] * u[1] * u[2] - xi9[1] * u[5] * u[7]

    # return [RHS[0], RHS[1], RHS[2], RHS[3], RHS[4], RHS[5], RHS[6], RHS[7], RHS[8] ]
    return RHS

def MFE_data_generation(Lx= 4*np.pi,Lz= 2*np.pi,Re=600, dt = 0.25, Tmax = 5000., plotting=0):
    """Function for the 2D data generation of the MFE system

    :param Lx: Domain size in x direction
    :param Lz: Domain size in z direction
    :param Re: Reynolds number of the flow
    :param dt: time step size
    :param Tmax: stopping time (assuming starting time is 0)
    :param plotting: bool property defining whether to plot the results
    :return: saves parameters, together with calculated Fourier coefficients u to hf file, no direct return
    """
    fln = 'MFE_Re' + str(int(Re)) + '_T' + str(int(Tmax)) + '.h5' # file name

    alpha = 2. * np.pi / Lx
    beta = np.pi / 2.
    gamma = 2. * np.pi / Lz

    zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9 = get_MFE_param(alpha, beta, gamma, Re)

    Nt = int(Tmax / dt)
    # EI = np.sqrt(1.1 - 1.) / 2.

    # values from Joglekar, Deudel & Yorke, "Geometry of the edge of chaos in a low dimensional turbulent shear layer model", PRE 91, 052903 (2015)
    u0 = np.array([1.0, 0.07066, -0.07076, 0.0 + 0.001 * np.random.rand(), 0.0, 0.0, 0.0, 0.0, 0.0])

    t = np.linspace(0, Tmax, Nt)

    # Energy0 = (1. - u0[0]) ** 2. + np.sum(u0[1:] ** 2.)
    # print(Energy0)

    u = np.zeros((Nt, 9))
    u[0, :] = u0

    for i in range(Nt - 1):
        # RHS = MFE_RHS(u[i,:], zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
        # u[i+1,:] = u[i,:] + dt*RHS
        k1 = dt * MFE_RHS(u[i, :], zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)
        k2 = dt * MFE_RHS(u[i, :] + k1 / 3., zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)
        k3 = dt * MFE_RHS(u[i, :] - k1 / 3. + k2, zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)
        k4 = dt * MFE_RHS(u[i, :] + k1 - k2 + k3, zeta, xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9)

        u[i + 1, :] = u[i, :] + (k1 + 3. * k2 + 3. * k3 + k4) / 8.

    # "turbulent energy"
    Energy = (1. - u[:, 0]) ** 2. + np.sum(u[:, 1:] ** 2., 1)

    f = h5py.File(fln, 'w')
    f.create_dataset('I', data=Energy)
    f.create_dataset('u', data=u)
    f.create_dataset('t', data=t)
    f.create_dataset('Re', data=Re)
    f.create_dataset('dt', data=dt)
    f.create_dataset('u0', data=u0)
    f.create_dataset('Lx', data=Lx)
    f.create_dataset('Lz', data=Lz)
    f.create_dataset('alpha', data=alpha)
    f.create_dataset('beta', data=beta)
    f.create_dataset('gamma', data=gamma)
    f.create_dataset('zeta', data=zeta)
    xiall = [xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9]

    for i in range(1, 10):
        var_name = 'xi' + str(i)
        f.create_dataset(var_name, data=xiall[i - 1])
    f.close()

    if plotting:
        plt.figure(figsize=(13, 7))
        plt.subplot(511)
        plt.plot(t, u[:, 0])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_1$")
        plt.subplot(512)
        plt.plot(t, u[:, 1])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_2$")
        plt.subplot(513)
        plt.plot(t, u[:, 2])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_3$")
        plt.subplot(514)
        plt.plot(t, u[:, 3])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_4$")
        plt.subplot(515)
        plt.plot(t, u[:, 4])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_5$")

        plt.figure(figsize=(13, 7))
        plt.subplot(511)
        plt.plot(t, u[:, 5])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_6$")
        plt.subplot(512)
        plt.plot(t, u[:, 6])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_7$")
        plt.subplot(513)
        plt.plot(t, u[:, 7])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_8$")
        plt.subplot(514)
        plt.plot(t, u[:, 8])
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("$u_9$")

        # plt.figure(3)
        plt.subplot(515)
        plt.plot(t, Energy)
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("t")
        plt.ylabel("E")

        plt.show()

    return 1

def MFE_read_Fourier(filename):
    """Function for reading MFE data including Fourier coefficients of the flow

    :param filename: name of hf file containing the data
    :return: returns time vector t and matrix of Fourier coefficients u (of size 9*Nt)
    """
    hf = h5py.File(filename, 'r')
    u = np.array(hf.get('/u'))
    t = np.array(hf.get('/t'))
    return t,u

def MFE_to_burst(u):
    """Function for translating Fourier coefficient data to mean shear, roll streak and burst components

    :param u: matrix of Fourier coefficients (of size Nt*9)
    :return: returns matrix of mean shear, roll streak and burst components x (of size Nt*3)
    """
    mean_shear = np.absolute(1-u[:,0])    # from a1,a9
    roll_streak = linalg.norm(u[:,1:3].transpose(),axis=0)  # from a2,a3,a4
    burst = linalg.norm(u[:,3:5].transpose(),axis=0)    # from a5,a6,a7,a8
    x = np.vstack([roll_streak, mean_shear, burst]).transpose()
    return x

def MFE_read_DI(filename, dt=0.25):
    """Function for reading MFE data including the dissipation and energy of the flow

    :param filename: part of name of .npy files containing the dissipation and energy data
    :param dt: time step
    :return: returns time vector t and matrix of dissipation and energy x (of size Nt*2)
    """
    D = np.load(filename+'_dissipation.npy')
    I = np.load(filename+'_energy.npy')
    t = np.arange(len(I))*dt
    x = np.append(D, I, axis=1)
    return t,x

plt.close('all') # close all open figures

# #####BURST#########
# type='MFE_burst'
# filename = 'MFE_Re400_T10000.h5'
# t,u = MFE_read_Fourier(filename)
#
# x = MFE_to_burst(u)
# dim = 3
# extr_dim = [2]   # define burst as the extreme dimension
#
# # Tesselation
# M = 20
#
# plotting = True
# min_clusters=30
# max_it=10
# clusters,P = extreme_event_identification_process(t,x,dim,M,extr_dim,type, min_clusters, max_it, 'classic', 7,plotting, False)
# plt.show()

####DISSIPATION#########
type='MFE_dissipation'
filename = 'MFE_Re600'
dt = 0.25
t,x = MFE_read_DI(filename, dt)
dim = 2
extr_dim = [0,1]    # define both dissipation and energy as the extreme dimensions

# Tesselation
M = 20

# plt.figure()
# plt.plot(t,x[:,0])
# plt.axhline(np.mean(x[:,0])+5*np.std(x[:,0]))
# plt.axhline(np.mean(x[:,0])+7*np.std(x[:,0]))
# plt.axhline(np.mean(x[:,0])+9*np.std(x[:,0]))
# plt.plot(t,x[:,1], 'r')
# plt.axhline(np.mean(x[:,1])+5*np.std(x[:,1]), color='red')
# plt.axhline(np.mean(x[:,1])+7*np.std(x[:,1]), color='red')
# plt.axhline(np.mean(x[:,1])+9*np.std(x[:,1]), color='red')

plotting = False
min_clusters=30 #20
max_it=10

t1 = time.time()
clusters, D, P = extreme_event_identification_process(t,x,dim,M,extr_dim,type, min_clusters, max_it, 'classic', 7,plotting, False)
elapsed = time.time() - t1
print("Time of clustering/deflating: ", elapsed)
extr_clusters = np.empty(0, int)
for i in range(len(clusters)):  #print average times spend in extreme clusters
    loc_cluster = clusters[i]
    if loc_cluster.is_extreme==2:
        extr_clusters = np.append(extr_clusters, i)
    # print("Average time in cluster ", loc_cluster.nr, " is: ", loc_cluster.avg_time, " s")


paths = find_extr_paths(extr_clusters,P)

min_prob = np.zeros((len(clusters),1))
min_time = np.zeros((len(clusters),1))

for i in range(len(clusters)):  # for each cluster
    # prob to extreme
    loc_prob,loc_time = prob_to_extreme(i, paths, t[-1], P, clusters)
    min_prob[i] = loc_prob
    min_time[i] = loc_time

# take (new) data series
t_new,x_new = MFE_read_DI(filename, dt)     #let's pretend this is a new data series

t2 = time.time()
#tesselate the new data
x_new_tess,temp= tesselate(x_new,M,[],7)    #tesselate function without extreme event id
x_new_tess = tess_to_lexi(x_new_tess, M, dim)

# cluster affiliation
x_new_clusters = data_to_clusters(x_new_tess, D)

elapsed = time.time() - t2
print("Time of analyzing new data: ", elapsed)

# print("Extreme event at t= ")
# show time series with extreme events (real-time)

fig, axs = plt.subplots(2)
fig.suptitle("Real-time predictions")

axs[0].set_xlim([t_new[0], t_new[-1]])
axs[0].set_xlabel("t")
axs[0].set_ylabel("D")

# axs[1].set_xlim([t_new[0], t_new[-1]])
axs[1].set_xlabel("t")
axs[1].set_ylabel("extreme")
axs[1].set_ylim([-0.5, 2.5])

n_skip=10
spacing = np.arange(0, len(t_new), n_skip, dtype=int)
for i in range(len(spacing)):
    if i!=0:
        loc_clust = data_to_clusters(x_new_tess[spacing[i]], D)
        axs[0].plot([t_new[spacing[i - 1]], t_new[spacing[i]]], [x_new[spacing[i - 1], 0], x_new[spacing[1], 0]], color='blue')

        temp2=clusters[x_new_clusters[spacing[i]]].is_extreme
        temp = clusters[x_new_clusters[spacing[i-1]]].is_extreme
        # probability of transitioning to extreme event (shortest path)    # minimum time to extreme event
        if temp2==2:
            axs[1].plot([t_new[spacing[i-1]], t_new[spacing[i]]], [temp, temp2],color='red')

        elif temp2==1:
            axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
                        [temp,
                         temp2], color='orange')
        else:
            if temp==2:  # if previous was extreme
                axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
                            [temp, temp2], color='red')
            elif temp==1:   # if previous was precursor
                axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
                            [temp, temp2], color='orange')
            else:
                axs[1].plot([t_new[spacing[i - 1]], t_new[spacing[i]]],
                        [temp, temp2], color='green')

        # text = fig.text(0.05, 0.03, str(clusters[x_new_clusters[spacing[i]]].prob_to_extreme))
        text = fig.text(0.05, 0.01, 'Probability: ' + str(min_prob[clusters[x_new_clusters[spacing[i]]].nr]))
        text2 = fig.text(0.05, 0.05, 'Time: ' + str(min_time[clusters[x_new_clusters[spacing[i]]].nr]))
        # axs[1].text(0,0,)
        axs[1].set_xlim([t_new[spacing[i-1]]-n_skip*10, t_new[spacing[i]]+ n_skip*10])
        plt.pause(0.001)
        text.remove()
        text2.remove()
plt.show()