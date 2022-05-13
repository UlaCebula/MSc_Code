# function for running the algorithm for the MFE system
# Urszula Golyska 2022

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import h5py
from my_func import *
import networkx as nx
from modularity_maximization import spectralopt
import scipy.sparse as sp
import seaborn as sns


def get_MFE_param(alpha, beta, gamma, Re):
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

    :param u:
    :param zeta:
    :param xi1-xi9:
    :return:
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

def data_generation_MFE(Lx= 4*np.pi,Lz= 2*np.pi,Re=600, dt = 0.25, Tmax = 5000., plotting=0):
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

def read_Fourier(filename):
    """Function for reading MFE data including Fourier coefficients of the flow

    :param filename: name of hf file containing the data
    :return: returns time vector t and matrix of Fourier coefficients u (of size 9*Nt)
    """
    hf = h5py.File(filename, 'r')
    u = np.array(hf.get('/u'))
    t = np.array(hf.get('/t'))
    return t,u

def to_burst(u):
    """Function for translating Fourier coefficient data to mean shear, roll streak and burst components

    :param u: matrix of Fourier coefficients (of size Nt*9)
    :return: returns matrix of mean shear, roll streak and burst components x (of size Nt*3)
    """
    mean_shear = np.absolute(1-u[:,0])    # from a1,a9
    roll_streak = linalg.norm(u[:,1:3].transpose(),axis=0)  # from a2,a3,a4
    burst = linalg.norm(u[:,3:5].transpose(),axis=0)    # from a5,a6,a7,a8
    x = np.vstack([roll_streak, mean_shear, burst]).transpose()
    return x

def read_DI(filename, dt=0.25):
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

def plot_time_series(x,t, type):
    """Function for plotting the time series of MFE data

    :param x: data matrix (look at to_burst or read_DI)
    :param t: time vector
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots time series of data (without plt.show())
    """
    if type=='burst':
        fig, axs = plt.subplots(5)
        fig.suptitle('Vertically stacked subplots')
        plt.subplot(5,1,1)
        plt.plot(t,u[:,0])
        plt.ylabel("$a_1$")
        plt.xlabel("t")
        plt.subplot(5,1,2)
        plt.plot(t,u[:,1])
        plt.ylabel("$a_2$")
        plt.xlabel("t")
        plt.subplot(5,1,3)
        plt.plot(t,u[:,2])
        plt.ylabel("$a_3$")
        plt.xlabel("t")
        plt.subplot(5,1,4)
        plt.plot(t,u[:,3])
        plt.ylabel("$a_4$")
        plt.xlabel("t")
        plt.subplot(5,1,5)
        plt.plot(t,u[:,4])
        plt.ylabel("$a_5$")
        plt.xlabel("t")

        fig, axs = plt.subplots(4)
        fig.suptitle('Vertically stacked subplots')
        plt.subplot(4,1,1)
        plt.plot(t,u[:,5])
        plt.ylabel("$a_6$")
        plt.xlabel("t")
        plt.subplot(4,1,2)
        plt.plot(t,u[:,6])
        plt.ylabel("$a_7$")
        plt.xlabel("t")
        plt.subplot(4,1,3)
        plt.plot(t,u[:,7])
        plt.ylabel("$a_8$")
        plt.xlabel("t")
        plt.subplot(4,1,4)
        plt.plot(t,u[:,8])
        plt.ylabel("$a_9$")
        plt.xlabel("t")

    if type=='dissipation':
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        plt.subplot(2, 1, 1)
        plt.plot(t, x[:, 0])
        plt.ylabel("D")
        plt.xlabel("t")
        plt.subplot(2, 1, 2)
        plt.plot(t, x[:, 1])
        plt.ylabel("I")
        plt.xlabel("t")

    return 1

def plot_phase_space(x, type):
    """Function for plotting the MFE data in phase space

    :param x: data matrix (look at to_burst or read_DI)
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots data x in equivalent phase space
    """
    if type=='burst':
        plt.figure()
        plt.plot(t, x[2,:])
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
        plt.xlabel("t")

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[0,:], x[1,:], x[2,:], lw=0.5)
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='dissipation':
        plt.figure()
        plt.plot(I, D)
        plt.title("Dissipation vs energy")
        plt.ylabel("D")
        plt.xlabel("I")
    return 1

def plot_tesselated_space(tess_ind,type):
    """Function for plotting the MFE data in tesselated phase space

    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots data x in equivalent tesselated phase space
    """
    if type=='burst':
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter3D(tess_ind[:, 0], tess_ind[:, 1], tess_ind[:, 2])
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")

    if type=='dissipation':
        plt.figure(figsize=(7, 7))
        plt.scatter(tess_ind[:,0], tess_ind[:,1], s=200, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("I")
        plt.ylabel("D")
    return 1

def plot_prob_matrix(P_dense):
    """Function for plotting probability matrix

    :param P_dense: dense representation of calculated probability matrix
    :return: none, plots probability matrix
    """
    # Visualize probability matrix
    plt.figure(figsize=(7, 7))
    plt.imshow(P_dense,interpolation='none', cmap='binary')
    plt.colorbar()
    return 1

def plot_graph(P_graph):
    """Function for plotting the graph representation of the probability matrix

    :param P_graph: graph form of the probability matrix
    :return: none, plots graph representation of probability matrix
    """
    # Visualize unclustered graph
    plt.figure()
    nx.draw_kamada_kawai(P_graph,with_labels=True)
    return 1

def plot_phase_space_clustered(x,type,D_nodes_in_clusters,tess_ind_cluster ):
    """Function for plotting phase space with cluster affiliation

    :param x: data matrix (look at to_burst or read_DI)
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :param D_nodes_in_clusters: matrix of affiliation of point to the community clusters
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :return: none, plots the phase space colored by cluster affiliation
    """
    if type=='burst':
        plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(D_nodes_in_clusters.shape[1]):   # for all communities
            ax.scatter(x[tess_ind_cluster==i,0], x[tess_ind_cluster==i,1], x[tess_ind_cluster==i,2])  # I should relate somehow s to N and the fig size
            x_mean = np.mean(x[tess_ind_cluster == i, 0])
            y_mean = np.mean(x[tess_ind_cluster == i, 1])
            z_mean = np.mean(x[tess_ind_cluster == i, 2])
            ax.text(x_mean, y_mean, z_mean, str(i))  # numbers of clusters
        ax.set_xlabel("Roll & streak")
        ax.set_ylabel("Mean shear")
        ax.set_zlabel("Burst")
        ax.set_title("Self-sustaining process")

    if type=='dissipation':
        plt.figure(figsize=(7, 7))
        # plt.plot(I,D, 'b--')
        for i in range(D_nodes_in_clusters.shape[1]):  # for all communities
            plt.scatter(x[tess_ind_cluster == i,1],
                        x[tess_ind_cluster == i,0])  # , c=palette[i])  # I should relate somehow s to N and the fig size
            x_mean = np.mean(x[tess_ind_cluster == i,1])
            y_mean = np.mean(x[tess_ind_cluster == i,0])
            plt.text(x_mean, y_mean, str(i))  # numbers of clusters
        plt.grid('minor', 'both')
        plt.minorticks_on()
        plt.xlabel("I")
        plt.ylabel("D")

    return 1

def plot_time_series_clustered(y,t, tess_ind_cluster, palette, type):
    """Function for plotting the time series of data with cluster affiliation

    :param y: vector of the parameter to plot; for type=="burst" y=burst; for type=="dissipation" y=D
    :param t: time vector
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param palette: color palette decoding a unique color code for each cluster
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots time series with cluster affiliation
    """
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i]!=tess_ind_cluster[i+1]:
            loc_col = palette[tess_ind_cluster[i]]
            plt.axvline(x=(t[i] + t[i + 1]) / 2, color=loc_col, linestyle='--')
            plt.text(t[i], 0.2, str(tess_ind_cluster[i]), rotation=90)  # numbers of clusters
    if type=='burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='dissipation':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    plt.xlabel("t")

    return 1

def plot_time_series_extr_iden(y,t, tess_ind_cluster, from_cluster, extr_cluster, type):
    '''Function for plotting time series with extreme event and precursor identification

    :param y: vector of the parameter to plot; for type=="burst" y=burst; for type=="dissipation" y=D
    :param t: time vector
    :param tess_ind_cluster: vector of the time series with their cluster affiliation
    :param from_cluster: vector of cluster numbers which can transition to extreme event
    :param extr_cluster: vector of cluster numbers which contain extreme events
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, plots time series with extreme event (blue) and precursor (red) identification
    '''
    plt.figure()
    plt.plot(t, y)
    for i in range(len(tess_ind_cluster)-1):
        if tess_ind_cluster[i] in from_cluster and tess_ind_cluster[i+1] in extr_cluster:
            plt.scatter(t[i], y[i], marker='s', facecolors = 'None', edgecolor = 'blue')
            plt.scatter(t[i+1], y[i+1], marker='s', facecolors='None', edgecolor='red')
    if type=='burst':
        plt.title("Burst component vs time")
        plt.ylabel("$b$")
    if type=='dissipation':
        plt.title("Dissipation vs time")
        plt.ylabel("$D$")
    plt.xlabel("t")
    return 1

def MFE_process(t,x,dim,M,extr_dim,type):
    """Big loop with calculation for the MFE system enclosed

    :param t: time vector
    :param x: data matrix (look at to_burst or read_DI)
    :param dim: integer, number of dimensions of the system (2 for type=="dissipation" and 3 for type=="burst")
    :param M: number of tesselation discretisations per dimension
    :param extr_dim: dimension which amplitude will define the extreme event (0 for type=="dissipation" and 2 for type=="burst")
    :param type: string defining the type of analysis, either "burst" or "dissipation"
    :return: none, runs calculations and plots results; can be modified to output the final deflated probability matrix
    """
    tess_ind, extr_id = tesselate(x, M, extr_dim)  # where extr_dim indicates the dimension by which the extreme event should be identified
    # Transition probability
    P = probability(tess_ind, 'classic')  # create sparse transition probability matrix

    tess_ind_trans = tess_to_lexi(tess_ind, M, dim)
    P, extr_trans = prob_to_sparse(P, M, extr_id)  # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

    # Graph form
    P_graph = to_graph_sparse(P)  # translate to dict readable for partition

    # Visualize unclustered graph
    plot_graph(P_graph)

    # Visualize probability matrix
    plot_prob_matrix(P.toarray())

    # Clustering
    P_community = spectralopt.partition(P_graph, refine=False)  # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
    D_sparse = community_aff_sparse(0, P_community, M, dim, 'first', 1)  # matrix of point-to-cluster affiliation

    # Deflate the Markov matrix
    P1 = sp.coo_matrix((D_sparse.transpose() * P) * D_sparse)
    print(np.sum(P1,axis=0).tolist())  # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P1_graph = to_graph(P1.toarray())

    # more iterations
    P_community_old = P_community
    P_old = P1
    P_graph_old = P1_graph
    D_nodes_in_clusters = D_sparse
    int_id = 0

    while int(np.size(np.unique(np.array(list(P_community_old.values()))))) > 20 and int_id < 10:  # condition
        int_id = int_id + 1
        P_community_old, P_graph_old, P_old, D_nodes_in_clusters = clustering_loop_sparse(P_community_old, P_graph_old,
                                                                                          P_old, D_nodes_in_clusters)
        # print(np.sum(D_nodes_in_clusters,axis=0).tolist())

    # Visualize clustered graph
    plot_graph(P_graph_old)

    # color palette
    palette = sns.color_palette(None, D_nodes_in_clusters.shape[1])

    # translate datapoints to cluster number affiliation
    tess_ind_cluster = data_to_clusters(tess_ind_trans, D_nodes_in_clusters)

    # identify extreme clusters and those transitioning to them
    extr_cluster, from_cluster = extr_iden(extr_trans, D_nodes_in_clusters, P_old)
    print('From cluster: ', from_cluster, 'To extreme cluster: ', extr_cluster)

    # Plot time series with clusters
    if type=='burst':
        plot_time_series_clustered(x[:,2], t, tess_ind_cluster, palette, type)
    if type=='dissipation':
        plot_time_series_clustered(x[:,0], t, tess_ind_cluster, palette, type)

    # Visualize phase space trajectory with clusters
    plot_phase_space_clustered(x, type, D_nodes_in_clusters, tess_ind_cluster)

    # Plot time series with extreme event identification
    if type == 'burst':
        plot_time_series_extr_iden(x[:,2], t, tess_ind_cluster, from_cluster, extr_cluster, type)
    if type == 'dissipation':
        plot_time_series_extr_iden(x[:,0], t, tess_ind_cluster, from_cluster, extr_cluster, type)

    # Visualize probability matrix
    plot_prob_matrix(P_old.toarray())

    return 1