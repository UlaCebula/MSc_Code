# testing the created tesselation and transition probability matrix functions on a simple sine wave case
# Urszula Golyska 2022

from my_func import *

def sine_data_generation(t0, tf, dt, nt_ex, rand_threshold=0.9, rand_amplitude=2, rand_scalar=1):
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

plt.close('all') # close all open figures
type='sine'

# time discretization
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex = 50  # number of time steps of the extreme event

# extreme event parameters
rand_threshold = 0.9
rand_amplitude = 2
rand_scalar = 1

t, x = sine_data_generation(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar)

dim = 2
extr_dim = [0,1]   # define both phase space coordinates as extreme event

# Tesselation
M = 20

plotting = True
min_clusters=15
max_it=5
clusters, D, P = extreme_event_identification_process(t,x,dim,M,extr_dim,type, min_clusters, max_it, 'classic', 2,plotting, False)

extr_clusters = np.empty(0, int)
for i in range(len(clusters)):  #print average times spend in extreme clusters
    loc_cluster = clusters[i]
    if loc_cluster.is_extreme==2:
        extr_clusters = np.append(extr_clusters, i)
    # print("Average time in cluster ", loc_cluster.nr, " is: ", loc_cluster.avg_time, " s")

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

plot_cluster_statistics(clusters, tf, min_prob, min_time, length)
plt.show()
