# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 06:48:29 2024

@author: frederic.ros
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from os import listdir
from os.path import isfile, join
import csv
import time



def scalar(x):
    m = np.mean(x,axis=0)
    s = np.std(x, axis=0)
    x = x - m
    x = x / s
    return x

def drawcluster2D3DConc(data, y,titre=0):
    fig = plt.figure()
    #plt.scatter(data[:, 0],data[:, 1], c=y, alpha=1, s=10)
    
    if len(data[0,:])>2:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=y, alpha=1, s=10)
    else:
        plt.scatter(data[:, 0],data[:, 1], c=y, alpha=1, s=10)
        
    plt.title(titre)
    plt.show()


def dist(data, centers):
    distance = np.sum((np.array(centers) - data[:, None, :])**2, axis = 2)
    return distance


def kmeans_plus_plus(X, k, pdf_method = True):
    '''Initialize one point at random.
    loop for k - 1 iterations:
        Next, calculate for each point the distance of the point from its nearest center. Sample a point with a 
        probability proportional to the square of the distance of the point from its nearest center.'''
    centers = []
    X = np.array(X)
    
    # Sample the first point
    initial_index = np.random.choice(range(X.shape[0]), )
    centers.append(X[initial_index, :].tolist())
    
#    print('max: ', np.max(np.sum((X - np.array(centers))**2)))
    
    # Loop and select the remaining points
    for i in range(k - 1):
        distance = dist(X, np.array(centers))
        
        if i == 0:
            pdf = distance/np.sum(distance)
            centroid_new = X[np.random.choice(range(X.shape[0]), replace = False, p = pdf.flatten())]
        else:
            # Calculate the distance of each point from its nearest centroid
            dist_min = np.min(distance, axis = 1)
            if pdf_method == True:
                pdf = dist_min/np.sum(dist_min)
                # Sample one point from the given distribution
                centroid_new = X[np.random.choice(range(X.shape[0]), replace = False, p = pdf)]
            else:
               index_max = np.argmax(dist_min, axis = 0)
               centroid_new = X[index_max, :]
        
        centers.append(centroid_new.tolist())
        
    return np.array(centers)

def get_random_centroids(data, k = 3):
    
    #return random samples from the dataset
    cent=[]
    for i in range(0,k):
        x = random.randint(0,len(data)-1)
        cent.append(data[x])
    
    return np.array(cent)


def GetG(X, label):
    
    ncluster = max(label) + 1
    occ = np.zeros(ncluster, int)
    G = np.zeros((ncluster, X.shape[1]), float)
    for i in range(0,ncluster):
        for j in range(0,X.shape[0]):
            if label[j] == i:
                G[i] = G[i] + X[j]
                occ[i] = occ[i] + 1
                
    for i in range(0,ncluster):
        G[i] = G[i] / occ[i]

    return G

def F_writefile(X,file_name):
    
    fw = open(file_name, 'w')
#    fichier.close()
    
#    with open(file_name, 'a') as fw:
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fw.write (str(X[i][j])+'\t')
        fw.write ('\n')
    fw.close()       
    return

def standardscalar(data):
    
    for i in range(0,data.shape[1]):
        V = data[:, [i]]
        m = np.mean(V)
        e = np.std(V)
        if e!=0:
            data[:, [i]] = (data[:, [i]] - m) / e
        else: data[:, [i]] = (data[:, [i]] - m)
    
    return data
    
def bugsil(score):
    
    score_true = []
    for i in range(0,len(score)):
        if score[i] <= 1 and score[i] >= -1:
            score_true.append(score[i])
   
    return score_true

def classif(liste1, liste2):
    
    if len(liste1) == 0: return 0
    s = 0
    for i in range(0,len(liste1)):
        if liste1[i] == liste2[i]:
            s = s + 1
    return 100 * (s/len(liste1))
#/////////////////////////////////////////////////////////
    
def getmergenoise(taille_init,liste_b1, liste_b2):
 
    liste_out=[]
    t = np.zeros(taille_init,(int))
    
    for i in range(0,len(liste_b1)):
        t[liste_b1[i]]=1
    
    for i in range(0,len(liste_b2)):
        t[liste_b2[i]]=1
    
    liste_bruit = []
    for i in range(0,taille_init):
        if t[i] == 1: liste_bruit.append(i)
           
    return liste_bruit
#/////////////////////////////////////////////////////////

def getlistwithoutnoise(liste_whole,liste_bruit):
    z = len(liste_whole)
    liste_out=[]
    if z==0: return 0
    
    t = np.ones(len(liste_whole),(int))
    
    for i in range(0,len(liste_bruit)):
        t[liste_bruit[i]]=0
        
    for i in range(0,len(liste_whole)):   
        if t[i]==1:    # c'est du bruit...
            X = liste_whole[i]
            liste_out.append(X)

    return liste_out #la liste bruit....

#/////////////////////////////////////////////////////////

def getNclusterfromcluster(cluster, seuil):
    n= len(cluster)
    s=0
    nc = 0
    for i in range(0,n):
       s = s + len(cluster[i])
    
    for i in range(0,n):
        if len(cluster[i])/s >= seuil:
            nc = nc + 1
    
    return nc

#/////////////////////////////////////////////////////////

def getNclusterfromlabel(label, seuil):
    n= max(label) + 1
    s = len(label)
    
    nc = 0
    for i in range(0,n):
        local = 0
        for j in range(0,s):
            if label[j] == i: local = local + 1
            
        if local/s >= seuil:
            nc = nc + 1
    
    return nc

#/////////////////////////////////////////////////////////
def getlabelfromclusters(cluster):
    
    n= len(cluster)
    s=0
    for i in range(0,n):
       s = s + len(cluster[i])
       
    label = np.zeros(s,(int))
    
    for i in range(0,n):
        for j in range(0, len(cluster[i])):
            label[cluster[i][j]] = i
            
    return label

#/////////////////////////////////////////////////////////
def getclusterfromlabel(label):
    ncluster = max(label) + 1
    
    cluster = []
    
    for i in range(0,ncluster):
        cluster_l = []
        for j in range(0,len(label)):
            if label[j] == i:
                cluster_l.append(j)
        cluster.append(cluster_l)
    
    return cluster



#/////////////////////////////////////////////////////////
def Plotdata2D(data, center, poids=[], titre='original data'):
     
    for i in range(0,data.shape[0]):
        plt.plot(data[i][0], data[i][1], 'bo')
    
    if poids==[]:
        for i in range(0,center.shape[0]):
            plt.plot(center[i][0], center[i][1],'bo',color='red', marker='x')
    else:
        for i in range(0,center.shape[0]):
            if poids[i]==1:
                plt.plot(center[i][0], center[i][1],'bo',color='red', marker='x')
#            else:
#                plt.plot(center[i][0], center[i][1],'bo',color='orange', marker='x')
            
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titre)
    plt.show()
    
#/////////////////////////////////////////////////////////      
def SinglePlotdata2D(data,  titre='original data'):
     
    for i in range(0,data.shape[0]):
        plt.plot(data[i][0], data[i][1], 'bo')
          
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titre)
    plt.show()

#////////////////////////////////////////////////////////////////////
def F_ReadfileWithClass (name,delim):
  
  X = 0
  fichier = open(name, "rt" )
  lecteurCSV = csv.reader(fichier,delimiter = delim)	# Ouverture du lecteur CSV en lui fournissant le caractère séparateur (ici ";")
  n = 0
  pold= 0
  Error = False
  for line in lecteurCSV:
    p = len(line)
    pold = p
    if n > 1 and pold != p:
      Error = True
    n += 1
  fichier.close()
#  p = p - 1
  
  if Error == False:
    fichier = open(name, "rt" )
    X = np.zeros((n,p),float)

    lecteurCSV = csv.reader(fichier,delimiter = delim)	# Ouverture du lecteur CSV en lui fournissant le caractère séparateur (ici ";")   
    n=0
    for line in lecteurCSV:
        for i in range(0,p):
            X[n,i]=line[i] 
#        print(X[n])    
        n += 1
    fichier.close()
        
    if X.shape[1]<2:# dimension > 2
        Error = False
        
  return Error,X
#////////////////////////////////////////////////////////////////////
def GetPartitionClasse(labels):
    nbclasse=0
    
    labels = np.asarray(labels)
    n = len(labels)
    occurence = max(labels) + 1    
    nbclasse= occurence - 2 #on enleve les bruits les frontieres...
        

    liste_bruit = []
    for k in range(0,n):
        if (labels[k] == 0 or labels[k] == 1): #on enleve les bruits et les frontieres
            liste_bruit.append(k)
            
    return nbclasse, liste_bruit

#////////////////////////////////////////////////////////////////////
def correctionlabel(label):
    c = max(label)
    ok = 0
    for i in range(0, len(label)):
        if label[i]==0: ok = 1
    if ok == 0:
        label = label - 1
    return label

#////////////////////////////////////////////////////////////////////
def lecturefile(title):
    data=y=0
    if title.endswith('.csv'):
        Y = pd.read_csv(title, sep=',')
    #    print("YSHAPE",Y.shape)
      #  print("Shape", Y.shape)
        data = Y.iloc[:, :-1].values
        data = np.asarray(data)
        categories = Y.iloc[:, -1].values
        y = pd.factorize(categories)[0]
        y = correctionlabel(y)

        #print(y)
        #y = categories
       # print(data)
#        labelencoder_y = LabelEncoder()
#        y = labelencoder_y.fit_transform(categories) #y contient les labels.
    else:
        Error,Y = F_ReadfileWithClass (title,'\t')
        if Error==True:
            print("reader problem")
        else:
            print("READ",Y.shape)
            y=np.zeros(Y.shape[0],int)
            data = Y[:, 0:(Y.shape[1]-1)]
            col = Y[:,(Y.shape[1]-1)] # returns the last columm
            for i in range(0,Y.shape[0]):
                y[i] = col[i]
            y = correctionlabel(y)
            #print(y)
#            Encoder = LabelEncoder()
#            y = Encoder.fit_transform(y)#on encode...    
              
    return data,y
