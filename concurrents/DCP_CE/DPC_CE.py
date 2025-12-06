# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:16:49 2024

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


def judge_nei(pair, dist, count, dc, dist_ther):
    turn = 0
    find_time = 0
    nei_all = [pair[0]]
    nei_new = nei_all.copy()
    nei_dist = 0
    
    while turn == 0:
        find_time += 1
        nei_temp = []
        
        if dist[pair[0], pair[1]] < dc * 2:
            turn = 1
            count[0] += 1
        
        for k in range(len(nei_new)):
            nei_add = np.where(dist[nei_new[k], :] < dist_ther)[0]
            nei_dist_max = np.mean(dist[nei_new[k], nei_add])
            if nei_dist_max > nei_dist:
                nei_dist = nei_dist_max
            nei_temp.extend(nei_add.tolist())
        
        nei_new = list(set(nei_temp) - set(nei_all))
        
        if pair[1] in nei_all:  # Check if pair[2] is in nei_all
            turn = 2
            count[1] += 1
            
        elif len(nei_new) < 1:  # Check if there are no new neighbors
            turn = 3
            count[2] += 1
        
        nei_all.extend(nei_new)
    
    return turn, nei_all, find_time, count, nei_dist


# Load data
# data_set = np.loadtxt('twomoons.txt')
data_set = np.loadtxt('fourlines.txt')

# Start timer
import time
start_time = time.time()

# Extract data and true labels
data = data_set[:, :-1]
true_label = data_set[:, -1]

# Centralize and scale the data
data = data - np.mean(data, axis=0)
data = data / np.max(np.abs(data))

# Plot :
if len(data[0,:])>2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=true_label, alpha=1, s=10)
else:
    plt.scatter(data[:, 0],data[:, 1], c=true_label, alpha=1, s=20)
plt.title('Data with true label', fontsize=12)

# Compute distance matrix if it doesn't exist
dim = data.shape[1]
num = data.shape[0]

mdist = []

for i in range(num):
    mdist_i = []
    for j in range(i + 1, num):
        mdist_i.append([i, j, np.linalg.norm(data[i] - data[j])])
    mdist.extend(mdist_i)

xx = np.array(mdist)

ND = int(np.max(xx[:, 1])+1)
NL = int(np.max(xx[:, 0])+1)
if NL > ND:
    ND = NL

N = xx.shape[0]

# Matrix of distance ND*ND
dist = np.zeros((ND, ND))

# Matrix of distance ND*ND, symmetry
for i in range(N):
    ii = int(xx[i, 0])
    jj = int(xx[i, 1])
    dist[ii, jj] = xx[i, 2]
    dist[jj, ii] = xx[i, 2]

# Compute dc: 2% dist
percent = 2.0
print(f'average percentage of neighbours (hard coded): {percent:5.6f}')

position = int(round(N * percent / 100))
sda = np.sort(xx[:, 2])
dc = sda[position]

print(f'Computing Rho with gaussian kernel of radius: {dc:12.6f}')

# Initialize rho: density of points
rho = np.zeros(ND)

# Gaussian kernel within dc and cut_off
for i in range(ND - 1):
    for j in range(i + 1, ND):
        if dist[i, j] < dc:
            rho[i] += np.exp(-(dist[i, j] / dc) ** 2)
            rho[j] += np.exp(-(dist[i, j] / dc) ** 2)

# Find the max distance
maxd = np.max(dist)

# Rank rho by descend
rho_sorted = np.sort(rho)[::-1]
ordrho = np.argsort(rho)[::-1]

# Deal with point with max rho
delta = np.zeros(ND)
nneigh = np.zeros(ND, dtype=int)
delta[ordrho[0]] = -1.
nneigh[ordrho[0]] = 0

# Compute the delta (relative distance), find nneigh for points
for ii in range(2, ND):
    delta[ordrho[ii]] = maxd
    for jj in range(1, ii):
        if dist[ordrho[ii], ordrho[jj]] < delta[ordrho[ii]]:
            delta[ordrho[ii]] = dist[ordrho[ii], ordrho[jj]]
            nneigh[ordrho[ii]] = ordrho[jj]

# Give max rho point max delta
delta[ordrho[0]] = np.max(delta)

# CES
count = np.zeros(3)
choose_num = 20
dist_ther1_ratio = 0.25
punish_ratio = 0.3

delta_sorted = np.sort(delta)[::-1]
orddelta = np.argsort(delta)[::-1]
choose_point = orddelta[:choose_num]
rho_choose_point = rho[choose_point]
ordrho_choose = np.argsort(rho_choose_point)[::-1]
pair_all = choose_point[ordrho_choose]
condition = []

for i in range(2, choose_num):
    pair_use = []
    for j in range(1, i + 1):
        if j == 1:
            pair = [pair_all[i], nneigh[pair_all[i]]]
            pair_use.append(nneigh[pair_all[i]])
        else:
            if pair_all[j - 1] in pair_use:
                continue
            else:
                pair = [pair_all[i], pair_all[j - 1]]
                pair_use.append(pair_all[j - 1])

        dist_ther1 = dist[pair[0], pair[1]]
        dist_ther = dist_ther1 * dist_ther1_ratio
        turn, nei_all, find_time, count, nei_dist = judge_nei(pair, dist, count, dc, dist_ther)

        if turn == 2:
            punish_time = find_time - 5
            dist_new = nei_dist + dist_ther * (1 + punish_time * punish_ratio)
            dist[pair[0], pair[1]] = dist_new
            dist[pair[1], pair[0]] = dist_new
            if dist_new < delta[pair_all[i]]:
                delta[pair_all[i]] = dist_new
                nneigh[pair_all[i]] = pair[1]

        if turn == 3 and j == 1:
            max_dist_nei = np.max(dist[pair[1], nei_all])
            dist[pair[0], pair[1]] = max_dist_nei * 1.1
            dist[pair[1], pair[0]] = dist[pair[0], pair[1]]
            delta[pair_all[i]] = dist[pair[0], pair[1]]

        condition1 = [pair, turn, find_time, nei_dist, dist_ther1, dist[pair[0], pair[1]]]
        condition.append(condition1)

maxd = np.max(dist)
delta[ordrho[0]] = np.max(delta)

# Decision graph
print('Generated file: DECISION GRAPH')
print('column 1: Density')
print('column 2: Delta')

with open('DECISION_GRAPH', 'w') as fid:
    for i in range(ND):
        fid.write(f'{rho[i]:6.2f} {delta[i]:6.2f}\n')

# Select a rectangle enclosing cluster centers
print('Select a rectangle enclosing cluster centers')

plt.figure()
plt.plot(rho, delta, 'o', markersize=5, markerfacecolor='k', markeredgecolor='k')
plt.title('Decision Graph of DPC-CE', fontsize=12)
plt.xlabel('ρ', fontsize=14)
plt.ylabel('δ', fontsize=14)

"""
rect = plt.ginput(1)
rhomin = rect[0][0]
deltamin = rect[0][1]

print('rhomin = ',rhomin)
print('deltamin = ', deltamin)
"""
rhomin = 2
deltamin = 0.5

# Initialize number of clusters
NCLUST = 0

# Clustering
cl = -np.ones(ND, dtype=int)
icl = np.zeros(ND, dtype=int)

for i in range(ND):
    if rho[i] > rhomin and delta[i] > deltamin:
        NCLUST += 1
        cl[i] = NCLUST
        icl[NCLUST - 1] = i

print(f'NUMBER OF CLUSTERS: {NCLUST}')

# Assign non-center points
for i in range(ND):
    if cl[ordrho[i]] == -1:
        cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

# Deal with halo
halo = cl.copy()

if NCLUST > 1:
    bord_rho = np.zeros(NCLUST)

    for i in range(ND-1):
        for j in range(i, ND):
            if cl[i] != cl[j] and dist[i, j] <= dc:
                rho_aver = (rho[i] + rho[j]) / 2.
                if rho_aver > bord_rho[cl[i]-1]:
                    bord_rho[cl[i]-1] = rho_aver
                if rho_aver > bord_rho[cl[j]-1]:
                    bord_rho[cl[j]-1] = rho_aver

    for i in range(ND):
        if rho[i] < bord_rho[cl[i]-1]:
            halo[i] = 0

# Print cluster information
for i in range(NCLUST):
    nc = np.sum(cl == i)
    nh = np.sum(halo == i)
    print(f'CLUSTER: {i} CENTER: {icl[i]} ELEMENTS: {nc} CORE: {nh} HALO: {nc - nh}')

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

