# testing different variations of the simple sine wave case
# Urszula Golyska 2022
from sine_func import *

## Increased probability of transitioning to extreme event
# plt.close('all') # close all open figures
# t0 = 0.0
# tf = 1000.0
# dt = 0.01
# nt_ex = 50  # number of time steps of the extreme event
# rand_threshold_array = [0.7, 0.75, 0.9, 0.95, 0.99]
# rand_amplitude = 2
# rand_scalar = 1
# dim = 2     # number of dimensions
# N = 20  # number of discretizations in each dimension
#
# plotting=0 # do we want to plot everything (otherwise plots only the last plot with extreme event identification)
# for rand_threshold in rand_threshold_array:
#     sine_process(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar, N, dim, plotting)
#     plt.show()

## Increased duration of state being within the extreme event
# plt.close('all') # close all open figures
# t0 = 0.0
# tf = 1000.0
# dt = 0.01
# nt_ex_array = [454]  # number of time steps of the extreme event
# rand_threshold= 0.95
# rand_amplitude = 2
# rand_scalar = 1
# dim = 2     # number of dimensions
# N = 20  # number of discretizations in each dimension
#
# plotting=1 # do we want to plot everything (otherwise plots only the last plot with extreme event identification)
# for nt_ex in nt_ex_array:
#     sine_process(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar, N, dim, plotting)
#     plt.show()

## Increased distance (in phase-space) of the extreme event
# plt.close('all') # close all open figures
# t0 = 0.0
# tf = 1000.0
# dt = 0.01
# nt_ex = 50  # number of time steps of the extreme event
# rand_threshold= 0.95
# rand_amplitude_array = [1.2, 1.5, 1.8]
# rand_scalar = 1
# dim = 2     # number of dimensions
# N = 20  # number of discretizations in each dimension
#
# plotting=1 # do we want to plot everything (otherwise plots only the last plot with extreme event identification)
# for rand_amplitude in rand_amplitude_array:
#     sine_process(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar, N, dim, plotting)
#     plt.show()

## Increased size (in phase-space) of the extreme event
plt.close('all') # close all open figures
t0 = 0.0
tf = 1000.0
dt = 0.01
nt_ex = 50  # number of time steps of the extreme event
rand_threshold= 0.95
rand_amplitude_array = [1.2, 1.5, 1.8]
rand_scalar = 1
dim = 2     # number of dimensions
N = 20  # number of discretizations in each dimension

plotting=0 # do we want to plot everything (otherwise plots only the last plot with extreme event identification)
for rand_amplitude in rand_amplitude_array:
    sine_process(t0, tf, dt, nt_ex, rand_threshold, rand_amplitude, rand_scalar, N, dim, plotting)
    plt.show()