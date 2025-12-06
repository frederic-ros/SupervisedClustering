# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:31 2024

@author: riad9


This is a replication of code originally written in MATLAB 
From "https://github.com/WJ-Guo/DPC-CE.git"
DPC-CE (Density Peak Clustering with Connectivity Estimation，Knowledge-Based Systems, 2022)
"""



import numpy as np
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from DPC_CE_ALGO import dpc_ce
   
# Load data
#data_set = np.loadtxt('twomoons.txt')
data_set = np.loadtxt('fourlines.txt')

# Start timer
import time
start_time = time.time()

# Extract data and true labels
data = data_set[:, :-1]
true_label = data_set[:, -1]

# Plot :
if len(data[0,:])>2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=true_label, alpha=1, s=10)
else:
    plt.scatter(data[:, 0],data[:, 1], c=true_label, alpha=1, s=20)
plt.title('Data with true label', fontsize=12)

###############################################################################
#                             LA FONCTION
###############################################################################
#
cl = dpc_ce(data, rhomin = 2, deltamin = .5)
#
###############################################################################
###############################################################################

# Plot clustering results
# Final Plot :
plt.figure()
if len(data[0,:])>2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=cl, alpha=1, s=10)
else:
    plt.scatter(data[:, 0],data[:, 1], c=cl, alpha=1, s=20)
plt.title('Results of DPC-CE', fontsize=12)

# Calculate FMI, ARI, NMI
DPCFMI = fowlkes_mallows_score(np.array(true_label), np.array(cl))
print(f'FMI value of DPC : {DPCFMI}')

DPCARI = adjusted_rand_score(np.array(true_label), np.array(cl))
print(f'ARI value of DPC : {DPCARI}')

DPCNMI = normalized_mutual_info_score(np.array(true_label), np.array(cl))
print(f'NMI value of DPC : {DPCNMI}')

# End timer
end_time = time.time()
print(f'Elapsed time: {end_time - start_time:.2f} seconds')

