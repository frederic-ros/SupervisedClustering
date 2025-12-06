# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:11:45 2023

@author: riad
"""
import numpy as np
import random as Random
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
 

#-------------------------------------------------------------------------------
def Hypercube(d_space, scale):
    C = pow(2,d_space)
    Centers = np.zeros((C,d_space), float)
    for i in range(C): #on balaye les centres.
        for j in range(d_space):
            d = pow(2,j)
            quotient = i // d
            if quotient % 2 == 1:
                Centers[i][j] = scale 
            else:
                Centers[i][j] = -scale

    return Centers
#-------------------------------------------------------------------------------

def UniformNoise(np_noise, dim = 2, t = 1, noise_label = -1):
    X = np.random.uniform(-t, t, int(dim*np_noise))
    X_noise = np.reshape(X, (np_noise, dim))
    y_noise = np.asarray([noise_label] * np_noise)
    
    return X_noise, y_noise
#------------------------------------------------------------------------------- 
def GetDataSet(n_samples = 256, n_features = 0, n_centers = 0, 
                noise = False, p_noise = 0, scale = 0, prop = True):
  
    if n_features == 0:
        n_features = Random.randint(2, 10)

    max_cluster = pow(2,n_features)
    if max_cluster>10:
        max_cluster = 10

    if n_centers == 0:
      n_centers = Random.randint(2, max_cluster)
    
    if scale == 0:
      scale = Random.uniform(3,5)
    C = Hypercube(n_features, scale)
    C = shuffle(C)
    Centers = []

    for c in range(n_centers):
        Centers.append(C[c])
        
    if noise:
        n_noise = int(p_noise/100*n_samples)
    else:
        n_noise = 0

    if not prop:
      Z, y = make_blobs(n_samples = n_samples - n_noise, n_features = n_features, 
                    centers = Centers, shuffle = False)
    else:
      n_samples0 = n_samples - n_noise
      n_clusters = len(Centers)
      r = [Random.random() for i in range(n_clusters)]
      s = sum(r)
      r = [ i/s for i in r ]
      n_samples_clust = [ int(i*n_samples0) for i in r ] 
      print(n_samples_clust)
      Z, y = make_blobs(n_samples = n_samples_clust, n_features = n_features, 
                    centers = Centers, shuffle = False)
    if noise:
        n_noise = n_samples - sum(n_samples_clust)
        Extrem_value = max([abs(np.amin(Z)), np.amax(Z)])
        Z_noise, y_noise = UniformNoise(n_noise, dim = n_features, 
                                        t = Extrem_value, noise_label = -1)
        Z = np.concatenate((Z, Z_noise))
        y = np.concatenate((y, y_noise))

    Z, y = shuffle(Z, y)

    return Z, y
#-------------------------------------------------------------------------------

# Demo :
'''
data, y = GetDataSet(n_samples = 256, 
                     n_features = 3, # if n_features = 0 (default), n_features is random between 2 and 10
                     n_centers = 6, # if n_centers = 0 (default),  n_centers is random between 1 and 10
                     noise = False, 
                     p_noise = 10 # p_noise is the precentage of noise (label = -1)
                    )
# Plot :
if len(data[0,:])>2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=y, alpha=1, s=10)
else:
    plt.scatter(data[:, 0],data[:, 1], c=y, alpha=1, s=20)
        

print("n_clusters: ",len(np.unique(y))-1)# number of clusters without noise
print("Data dim (size, n_features): ", data.shape)'''