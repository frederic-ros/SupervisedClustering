# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 09:18:32 2024

@author: frederic.ros
"""
import numpy as np
from sklearn.cluster import KMeans

def launchKmeans(data, ncluster_min, ncluster_max):
    
    if ncluster_max - ncluster_min > 2:
        wcss = []
        for i in range(ncluster_min, ncluster_max + 1):
            mykmeans = KMeans(n_clusters = i, n_init = "auto", init = 'k-means++', random_state = 0)
            mykmeans.fit(data)
            wcss.append(mykmeans.inertia_)
        
    
        # Calculer les dérivées premières et secondes
        first_derivative = np.diff(wcss)
        second_derivative = np.diff(first_derivative)
    
        # Trouver le point du coude (max de la dérivée seconde)
        elbow_point = np.argmax(second_derivative) + 1 + ncluster_min  # +1 pour correspondre à l'index de K
    else:
        elbow_point = ncluster_min
        
    mykmeans = KMeans(n_clusters = elbow_point, n_init = "auto", init = 'k-means++', random_state = 0)
    y_kmeans = mykmeans.fit_predict(data)
    
    return y_kmeans
