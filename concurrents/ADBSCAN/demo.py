# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:31:59 2024

@author: riad9
"""

from sdbscan import SDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from DataSet import GetDataSet
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

n_samples = 1000

data, y_true = GetDataSet(n_samples = n_samples, 
                     n_features = 3, # if n_features = 0 (default), n_features is random between 2 and 10
                     n_centers = 3, # if n_centers = 0 (default),  n_centers is random between 1 and 10
                     noise = True, 
                     p_noise = 10 # p_noise is the precentage of noise (label = -1)
                    )
'''
if data.shape[1] > 8: 
        dataTsne = TSNE(n_components=8, learning_rate='auto',init='pca', 
                                  perplexity=3).fit_transform(data)
        data = dataTsne
'''       
if data.shape[1] > 8:
    pca = PCA(n_components=8)
    pcadata = pca.fit_transform(data)
    data = pcadata
    
min_samples = int(1*np.sqrt(n_samples))
min_samples = (int)(0.1 * n_samples)
labels = SDBSCAN(min_samples=min_samples,
                 noise_percent=0.2
                 ).fit_predict(data)



# Plot :
if len(data[0,:])>2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=labels, alpha=1, s=10)
else:
    plt.scatter(data[:, 0],data[:, 1], c=labels, alpha=1, s=20)
    
    

print('\nARI = ',metrics.adjusted_rand_score(y_true, labels))

print('\nAMI = ',metrics.adjusted_mutual_info_score(y_true, labels))

print("\nn_clusters: ",len(np.unique(labels))-1)# number of clusters without noise
print("\nData dim (size, n_features): ", data.shape)