# testing the created tesselation and transition probability matrix functions on a simple sine wave case
# Urszula Golyska 2022

from my_func import *
import csv

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

extr_dim = [0,1]   # define both phase space coordinates as extreme event

# Tesselation
M = 30

plotting = True
min_clusters=15
max_it=5
clusters, D, P = extreme_event_identification_process(t,x,M,extr_dim,type, min_clusters, max_it, 'classic', 2,plotting, True)
calculate_statistics(extr_dim, clusters, P, tf)
plt.show()

x_tess,temp = tesselate(x,M,extr_dim,7)    #tesselate function without extreme event id
x_tess = tess_to_lexi(x_tess, M, 2)
x_clusters = data_to_clusters(x_tess, D, x, clusters)
is_extreme = np.zeros_like(x_clusters)
for cluster in clusters:
    is_extreme[np.where(x_clusters==cluster.nr)]=cluster.is_extreme

avg_time, instances, instances_extreme_no_precursor, instances_precursor_no_extreme = backwards_avg_time_to_extreme(is_extreme,dt, clusters)
print('Average time from precursor to extreme:', avg_time, ' s')
print('Nr times when extreme event had a precursor:', instances)
print('Nr extreme events without precursors (false negative):', instances_extreme_no_precursor)
print('Percentage of false negatives:', instances_extreme_no_precursor/(instances+instances_extreme_no_precursor)*100, ' %')
print('Percentage of extreme events with precursor (correct positives):', instances/(instances+instances_extreme_no_precursor)*100, ' %')
print('Nr precursors without a following extreme event (false positives):', instances_precursor_no_extreme)
print('Percentage of false positives:', instances_precursor_no_extreme/(instances+instances_precursor_no_extreme)*100, ' %')

# colors = ['#1f77b4', '#ff7f0e', '#d62728']     # blue, orange, red
#
# fig, axs = plt.subplots(2)
# plt.subplot(2, 1, 1)
# plt.plot(t,x[:,0])
# plt.ylabel("D")
# plt.xlabel("t")
#
# plt.subplot(2, 1, 2)
# plt.plot(t,x[:,1])
# plt.ylabel("k")
# plt.xlabel("t")
#
# for i in range(len(t) - 1):
#     if is_extreme[i]!=is_extreme[i+1]:
#         loc_col=colors[is_extreme[i+1]]
#         plt.subplot(2, 1, 1)
#         plt.axvline(x=t[i+1], color=loc_col, linestyle='--')
#
#         plt.subplot(2, 1, 2)
#         plt.axvline(x=t[i + 1], color=loc_col, linestyle='--')
#
# plt.show()
