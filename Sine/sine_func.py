# function for running the algorithm for the sine wave
# Urszula Golyska 2022
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from my_func import *

def data_generation(t0=0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar):
    """Function for generating data of the sine wave system

    :param t0: starting time, default t0=0
    :param tf: final time
    :param dt: time step
    :param nt_ex: number of time steps that the extreme event will last
    :param rand_threshold: threshold defining the probability of the extreme event
    :param rand_amplitude: amplitude defining the jump of the extreme event
    :param rand_scalar: scalar value defining the size of the extreme event
    :return: returns time vector t and matrix of data (of size 2*Nt)
    """
    # time discretization
    t = np.arange(t0,tf,dt)

    # phase space
    u1 = np.sin(t)
    u2 = np.cos(t)

    # generate random spurs
    for i in range(len(t)):
        if u1[i]>=0.99 and abs(u2[i])<=0.015:
            if np.random.rand() >=rand_threshold:
                u1[i:i+nt_ex] = rand_scalar*u1[i:i+nt_ex]+rand_amplitude
                u2[i:i+nt_ex] = rand_scalar*u2[i:i+nt_ex]+rand_amplitude

    u = np.hstack([np.reshape(u1, (len(t),1)),np.reshape(u2, (len(t),1))])  # combine into one matrix
    return t, u

def plot_time_series(t,u):
    """Function for plotting the time series of the data

    :param t: time vector
    :param u: matrix of data (of size 2*Nt)
    :return: none, plots the time series of u (sin(t) and cos(t))
    """
    # Visualization of the two dimensions - time series
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    plt.subplot(2,1,1)
    plt.plot(t, u[:,0])
    plt.ylabel("$u_1$")
    plt.xlabel("t")
    plt.subplot(2,1,2)
    plt.plot(t, u[:,1])
    plt.ylabel("$u_2$")
    plt.xlabel("t")

    # Phase space u1,u2
    plt.figure(figsize=(7, 7))
    plt.plot(u[:,0],u[:,1])
    plt.grid('minor', 'both')
    plt.minorticks_on()
    plt.xlabel("$u_1$")
    plt.ylabel("$u_2$")
    return 1

def plot_tesselated_space(tess_ind):
    """Function for plotting the data in tesselated phase space

    :param tess_ind: matrix which includes the indices of the box taken by the data points in consequent time steps, can
    be obtained from the tesselate(x,N) function
    :return: none, plots the data in tesselated phase space
    """
    # Visualization - tesselated space
    plt.figure(figsize=(7, 7))
    plt.scatter(tess_ind[:,0], tess_ind[:,1], s=200, marker='s', facecolors = 'None', edgecolor = 'blue') #I should relate somehow s to N and the fig size
    plt.grid('minor', 'both')
    plt.minorticks_on()
    plt.xlabel("$u_1$")
    plt.ylabel("$u_2$")
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

def plot_extreme(t,u,tess_ind_trans,nodes_to,nodes_from):
    """Function for plotting time series with extreme event and precursor identification
    OLD, INEFFICIENT IMPLEMENTATION

    :param t: time vector
    :param u: matrix of data (of size 2*Nt)
    :param tess_ind_trans: datapoint time series already translated to tesselated lexicographic ordering
    :param nodes_to: vector of nodes - extreme events
    :param nodes_from: vector of nodes which can transition to extreme events
    :return: none, plots time series with extreme event (blue) and precursor (red) identification
    """

    # Identify where we have a from node followed by a to node
    # Visualization
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    plt.subplot(2,1,1)
    plt.plot(t, u[:,0])
    plt.ylabel("$u_1$")
    plt.xlabel("t")
    plt.subplot(2,1,2)
    plt.plot(t, u[:,1])
    plt.ylabel("$u_2$")
    plt.xlabel("t")

    # works but computationally inefficient
    temp_to = np.array(np.where(np.isin(tess_ind_trans, nodes_to)))
    temp_to=temp_to[0]
    for i in range(len(temp_to)):
        j = temp_to[i]
        if tess_ind_trans[j-1] in nodes_from:
                t_from = t[j-1]
                t_to = t[j]
                axs[0].scatter(t_from, u[j-1,0], marker='s', facecolors = 'None', edgecolor = 'blue')
                axs[0].scatter(t_to, u[j,0], marker='s', facecolors='None', edgecolor='red')
                axs[1].scatter(t_from, u[j-1,1], marker='s', facecolors='None', edgecolor='blue')
                axs[1].scatter(t_to, u[j,1], marker='s', facecolors='None', edgecolor='red')
    return 1

def sine_process(t0=0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar, N, dim=2, plotting):
    """Big loop with calculation for the sine wave system enclosed

    :param t0: starting time, default t0=0
    :param tf: final time
    :param dt: time step
    :param nt_ex: number of time steps that the extreme event will last
    :param rand_threshold: threshold defining the probability of the extreme event
    :param rand_amplitude: amplitude defining the jump of the extreme event
    :param rand_scalar: scalar value defining the size of the extreme event
    :param N: number of tesselation discretisations per dimension
    :param dim: number of dimensions, default 2
    :param plotting: bool property defining whether to plot the results
    :return: none, runs calculations and plots results; can be modified to output the final deflated probability matrix
    """
    t, u = data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)
    if plotting:
        plot_time_series(t, u)

    # Tesselation
    tess_ind, extr_id = tesselate(u,N,0)  # where 0 indicates the dimension by which the extreme event should be identified - here u1
    if plotting:
        plot_tesselated_space(tess_ind)

    # Transition probability matrix
    P = probability(tess_ind, 'classic')  # create sparse transition probability matrix

    tess_ind_trans = tess_to_lexi(tess_ind, N, dim)
    P, extr_trans = prob_to_sparse(P,N,extr_id)  # translate matrix into 2D sparse array with points in lexicographic order, translates extr_id to lexicographic id

    P_dense = P.toarray()
    if plotting:
        plot_prob_matrix(P_dense)

    # Graph form
    P_graph = to_graph(P_dense)  # translate to dict readable for partition function
    if plotting:
        plot_graph(P_graph)

    # Clustering
    P_community = spectralopt.partition(P_graph)  # partition community P, default with refinement; returns dict where nodes are keys and values are community indices
    D = community_aff(0, P_community, N, dim, 'first', 1)  # matrix of point-to-cluster affiliation

    # Deflate the Markov matrix
    P1 = np.matmul(np.matmul(D.transpose(), P_dense.transpose()), D)
    print(np.sum(P1,axis=1).tolist())  # should be approx.(up to rounding errors) equal to number of nodes in each cluster

    # Graph form
    P1 = P1.transpose()  # had to add transpose for the classic probability, why? the same for backwards?
    P1_graph = to_graph(P1)
    if plotting:
        plot_graph(P1_graph)

    # Identify extreme event it's precursor
    extreme_from, extreme_to, nodes_from, nodes_to = extr_iden(P1,P_community,'deviation',extr_trans)  # for bifurcation it detect's wrong path
    print('From: ', extreme_from, '(', nodes_from, ') \n', 'To: ', extreme_to, '(', nodes_to, ')')

    plot_extreme(t, u, tess_ind_trans, nodes_to, nodes_from)
    return 1