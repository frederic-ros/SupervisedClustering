# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 06:26:00 2024

@author: frederic.ros
"""

import numpy as np


#*****************************************************
def GetListeNoise(labels, label_noise):
    
    n = len(labels)
    liste_bruit = []
    for k in range(0,n):
        if (labels[k] == label_noise): #on enleve les bruits et les frontieres
            liste_bruit.append(k)
            
    return liste_bruit

#*****************************************************
def getDatawithout(data, liste_bruit):
    z = len(data)
    
    t = len(data) - len(liste_bruit)
    
    if z==0: return 0
    
    data_out=np.zeros((t, data.shape[1]), float)
    t = np.ones(len(data),(int))
    
    for i in range(0,len(liste_bruit)):
        t[liste_bruit[i]]=0
        
    u = 0    
    for i in range(0,len(data)):   
        if t[i]==1: 
            data_out[u] = data[i]
            u = u + 1
        
   
    return data_out

#*****************************************************
def getNovellistwithoutnoise(liste_whole,liste_bruit):

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
#*****************************************************
