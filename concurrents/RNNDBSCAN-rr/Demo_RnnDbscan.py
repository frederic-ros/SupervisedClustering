# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:56:21 2024

@author: riad9

"""


from scipy.spatial import distance
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from DataSet import GetDataSet




UNCLASSIFIED, NOISE, CORE, BOUNDARY = -1, -2, -3, -4
DataPoint = namedtuple('DataPoint', ['baseValues', 'rnnValues', 'knnValues', 'distanceValues',
                                     'index', 'classifier'])
X = []

"""
La démonstration  fonction main() 
Sinon dans dans un autre fichier il faut importer : 
    createDistanceMatrix, buildDataSet et RNN_DBSCAN
"""
def main():
    global X
    N_SAMPLES = 1000
    rangk = [15, 20, 25] # Il faut trouver la bonne valeur de k
    
    features, y_true = GetDataSet(n_samples = N_SAMPLES, 
                         n_features = 3, # if n_features = 0 (default), n_features is random between 2 and 10
                         n_centers = 4, # if n_centers = 0 (default),  n_centers is random between 1 and min(10,n_features^2)
                         noise = True, 
                         p_noise = 10 # p_noise is the precentage of noise (label = -1)
                        )
    
    # Centralize and scale features
    features = features - np.mean(features, axis=0)
    features = features / np.max(np.abs(features))
    classList = y_true
    
    distanceMatrix = createDistanceMatrix(features)
    
    for k in rangk:
        X = buildDataSet(k, features, classList, distanceMatrix)
        assign = RNN_DBSCAN(X, k)
        labels = np.array(assign)
        
        plt.figure()
        if len(features[0,:])>2:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(features[:, 0],features[:, 1],features[:, 2], c=labels, alpha=1, s=10)
        else:
            plt.scatter(features[:, 0],features[:, 1], c=labels, alpha=1, s=20)
        
        plt.show()
        
        print("\nk value:", k)
        print('ARI = ',metrics.adjusted_rand_score(y_true, labels))
    
        print('AMI = ',metrics.adjusted_mutual_info_score(y_true, labels))
    
        print("n_clusters: ",len(np.unique(labels))-1)# number of clusters without noise




def preprocess(data, splitOn):
    classes = 1
    features = []
    classDict = {}
    classList = []
    for line in data:
        line = line.split(splitOn)
        if len(line) > 1:
            features.append([float(i) for i in line[:-1]])
            if line[-1] not in classDict:
                classDict[line[-1]] = classes
                classes += 1
            classList.append(classDict[line[-1]])

    return features,classList,classDict


def createDistanceMatrix(data):
    outMatrix = [[-1 for y in data] for i in data]
    for baseIndex, baseRow in enumerate(data):
        for otherIndex, otherRow in enumerate(data[baseIndex+1:]):
            dst = distance.euclidean(baseRow,otherRow)
            outMatrix[baseIndex][otherIndex+1+baseIndex] = dst
            outMatrix[otherIndex+1+baseIndex][baseIndex] = dst
    return tuple([tuple(row) for row in outMatrix])


def KNN(distanceMatrix,k):
    k += 1
    newMatrix = [sorted([(index,value) for index,value in enumerate(i)], key=lambda x:x[1])[::1][:k] for i in distanceMatrix]
    newMatrix = [[index for index,dist in row] for row in newMatrix]
    newMatrix = [row[1:] for row in newMatrix]
    return tuple([tuple(row) for row in newMatrix])


def RNN(nm):
    R = [[] for i in nm]
    for base, row in enumerate(nm):
        for v in row:
            R[v].append(base)
    return tuple([tuple(row) for row in R])


def buildDataSet(k, data, classifierMatrix, distanceMatrix):
    kMatrix = KNN(distanceMatrix,k)
    rMatrix = RNN(kMatrix)
    return tuple([DataPoint(data[i],rMatrix[i],kMatrix[i],distanceMatrix[i],i,classifierMatrix[i]) for i in range(len(data))])


def RNN_DBSCAN(X, k):
    assign = [UNCLASSIFIED for i in range(len(X))]
    cluster = 1

    for i, x in enumerate(X):
        if assign[i] == UNCLASSIFIED:
            if expandCluster(x, cluster, assign, k):
                cluster += 1

    expandClusters(k,assign)

    return assign


def expandCluster(x, cluster, assign, k):
    if(len(x.rnnValues)) < k:
        assign[x.index] = NOISE
        return False

    else:
        seeds = neighborhood(x,k)
        for v in [x] + seeds: assign[v.index] = cluster

        while seeds:
            y = seeds.pop(0)
            if len(y.rnnValues) >= k:
                neighbors = neighborhood(y,k)
                for z in neighbors:
                    if assign[z.index] == UNCLASSIFIED:
                        seeds.append(z)
                        assign[z.index] = cluster
                    elif assign[z.index] == NOISE:
                        assign[z.index] = cluster
        return True


def neighborhood(x,k):
    return [X[val] for val in x.knnValues] + [X[y] for y in x.rnnValues if len(X[y].rnnValues) >= k]


def expandClusters(k,assign):
    for x in X:
        if assign[x.index] == NOISE:
            neighbors = x.knnValues
            mincluster = NOISE
            mindist = -1

            for i in neighbors:
                n = X[i]
                cluster = assign[i]
                d = x.distanceValues[i]

                if len(n.knnValues) >= k and d <= density(cluster,assign) and (d < mindist or mindist == -1):
                    mincluster = cluster
                    mindist = d
            assign[x.index] = mincluster


def density(cluster, assign):
    clusterPoints = [i for i in range(len(assign)) if assign[i] == cluster]
    maxDist = -1
    for baseIndex, baseRow in enumerate(clusterPoints):
        for otherIndex, otherRow in enumerate(clusterPoints[baseIndex+1:]):
            if maxDist < X[baseRow].distanceValues[otherRow]:
                maxDist = X[baseRow].distanceValues[otherRow]

    return maxDist

def rnndbscan(features, y_true, k):
    global X

#    rangk = [15, 20, 25] # Il faut trouver la bonne valeur de k
    
    '''
    # Centralize and scale features
    features = features - np.mean(features, axis=0)
    features = features / np.max(np.abs(features))
    '''
    classList = y_true
    distanceMatrix = createDistanceMatrix(features)
    
    X = buildDataSet(k, features, classList, distanceMatrix)
    assign = RNN_DBSCAN(X, k)
    labels = np.array(assign)
    '''    
    plt.figure()
    if len(features[0,:])>2:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(features[:, 0],features[:, 1],features[:, 2], c=labels, alpha=1, s=10)
    else:
        plt.scatter(features[:, 0],features[:, 1], c=labels, alpha=1, s=20)
        
    plt.show()
        
    print("\nk value:", k)
    print('ARI = ',metrics.adjusted_rand_score(y_true, labels))
    
    print('AMI = ',metrics.adjusted_mutual_info_score(y_true, labels))
    
    print("n_clusters: ",len(np.unique(labels))-1)# number of clusters without noise
    '''
    return labels
'''
if __name__ == '__main__':
    main()
'''