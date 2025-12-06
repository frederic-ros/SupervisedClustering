# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 06:15:34 2024

@author: frederic.ros
"""
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import csv
import time
from util import lecturefile, SinglePlotdata2D, GetPartitionClasse, getlabelfromclusters,getclusterfromlabel
from util import getlistwithoutnoise, getNclusterfromcluster, getNclusterfromlabel, classif, bugsil, standardscalar
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score
from util import drawcluster2D3DConc, scalar
from sklearn.manifold import TSNE
from sdbscan import SDBSCAN

#////////////////////////////////////////////////////////////////////
def Test_databaseADBSCAN(directoire_data, directoire_label, directoire_result, Usebruit=1, 
                        AvecLabel = 1, dessin=0):
    Usebruit = AvecLabel
    ok=0
    
    centre=[]
      
    truelabel=[]
    fichiers_data = [f for f in listdir(directoire_data) if isfile(join(directoire_data, f))]
    fichiers_label = [f for f in listdir(directoire_label) if isfile(join(directoire_label, f))]
    n = len(fichiers_data)
    l_base=[]
    Tsne = 1
    
   
    nCreel=[]
   
    Sil=[]
    Rand=[]           
    npred=[]
          
    ttime = []
    liste_bruitreel = []
    Aseuil = 1
    #n = 3
    for i in range(0,n):    
       #u = random.randint(0, len(fichiers_data)-1)
       u=i
       ok=0
       
       if (AvecLabel == 0):
           titlecomplet_data= directoire_data + "/"+ fichiers_data[u]
           
           if titlecomplet_data.endswith('.csv') or titlecomplet_data.endswith('.txt'):
               data, y = lecturefile(titlecomplet_data)
               data = standardscalar(data)
               if dessin==1: SinglePlotdata2D(data, titre='original data')
               truelabel = y
               truelabelnonoise = truelabel
               Nreel = max(truelabel) + 1
               ok=2
               
       if AvecLabel == 1:
           titlecomplet_data= directoire_data + "/"+ fichiers_data[u]
           titlecomplet_label= directoire_label + "/"+ fichiers_label[u]
        
           if titlecomplet_data.endswith('.csv'):
                data = pd.read_csv(titlecomplet_data,header=None)
                data = np.asarray(data)  
                data = scalar(data)
                #if (data.shape[1] != 2): continue
                if dessin==1: SinglePlotdata2D(data, titre='original data')
                ok=1
            
           if titlecomplet_label.endswith('.csv'):
                truelabel = pd.read_csv(titlecomplet_label,header=None)
                truelabel = np.asarray(truelabel)
                truelabel = np.reshape(truelabel,1000)
                c,liste_bruitreel = GetPartitionClasse(truelabel)
                Nreel = max(truelabel) + 1
                if (Usebruit == 1):
                    truelabelnonoise = getlistwithoutnoise(truelabel, liste_bruitreel)
                    print("Taille des true labels:",len(truelabelnonoise), len(liste_bruitreel))
                    truelabelnonoise = np.reshape(truelabelnonoise,len(truelabelnonoise))
                    Nreel = max(truelabel) - 1
                
            
                ok = ok + 1
            
           
       if (ok==2):
           
           print(fichiers_data[u], fichiers_label[u],"p:",data.shape[1])  
           a = time.perf_counter()
           #print("SHAPE",data.shape)
           
           min_samples = int(np.sqrt(data.shape[0]))
           predlabels = SDBSCAN(min_samples=min_samples,noise_percent=0.05).fit_predict(data)
          
           b = time.perf_counter()
           
           if dessin==1: drawcluster2D3DConc(data, predlabels, titre="clustering brut")
           if Usebruit == 1:
               predlabels = getlistwithoutnoise(predlabels, liste_bruitreel) #les labels sans le bruit.
               if dessin==1:
                   data_withoutnoise = data[truelabel>=2]
                   drawcluster2D3DConc(data_withoutnoise, predlabels, titre="clustering filtrage bruit")
               
           if Aseuil == 1:   ncS = getNclusterfromlabel(predlabels, 0.05)
           else:  ncS = max(predlabels)+1
    
           ncXmeans=ncS 
           
           if Usebruit == 1:
               #print("tailles:", len(predlabels), len(truelabelnonoise))
               RandXmeans = adjusted_rand_score(truelabelnonoise, predlabels)
               if ncXmeans > 1: 
                   if AvecLabel == 1: 
                       SilXmeans = silhouette_score(data[truelabel>=2], predlabels)
                       if Tsne == 1: 
                           dataTsne = TSNE(n_components=2, learning_rate='auto',init='pca', 
                                  perplexity=3).fit_transform(data_withoutnoise)
                           drawcluster2D3DConc(dataTsne, predlabels,titre=fichiers_data[u] )   

                   else:  
                       SilXmeans = silhouette_score(data[truelabel], predlabels)
               else: SilXmeans = 0
           else:
               RandXmeans = adjusted_rand_score(truelabel, predlabels)
               if ncXmeans > 1: 
                   SilXmeans = silhouette_score(data, predlabels)
                   if Tsne == 1: 
                       dataTsne = TSNE(n_components=2, learning_rate='auto',init='pca', 
                                  perplexity=3).fit_transform(data)
                       drawcluster2D3DConc(dataTsne, predlabels,titre=fichiers_data[u] )   

               else: SilXmeans = 0

           
           nCreel.append(Nreel)
           Sil.append(SilXmeans)
           Rand.append(RandXmeans)           
           npred.append(ncXmeans)
          
           print("Result(Cr,C,Sil,R)", Nreel, ncXmeans, SilXmeans, RandXmeans)
           
           ttime.append(b-a)
           
           
    a = classif(nCreel, npred)
    
    if AvecLabel == 1:
        return a, np.mean(Sil), np.mean(Rand)
    else:
        return npred, Sil, Rand
    
#    return 0
#    print("Moy time", np.mean(ttime), np.std(ttime))
    
#----------------------------------------------------------------------------------
def competiteursADBSCAN(directoire, AvecLabel, dessin):
    
    if AvecLabel == 1:
        directoire_data = directoire + "/data"
        directoire_label = directoire + "/label"
        directoire_result = directoire + "/result"
    else:
        directoire_data = directoire 
        directoire_label = directoire
        directoire_result = directoire
        
    print("directoire data", directoire_data)    
    
    X =  0
    
    X = Test_databaseADBSCAN(directoire_data, directoire_label, directoire_result, Usebruit=1, 
                        AvecLabel=AvecLabel, dessin=dessin)
    

 
    return X
#----------------------------------------------------------------------------------
#directoire = "..\Base3D" #0.2 et 0.05
#directoire = "..\Base7D" #0.2 et 0.05
#directoire = "..\Base1" #0.33 et 0.15
#directoire = "..\Base2" #0.3 et 0.1
#directoire = "..\Base3" #0.33 et 0.15
#directoire = "..\Realdatabase"
#directoire = "..\TestADBSCAN"
directoire = "..\TestForme"
X = competiteursADBSCAN(directoire, AvecLabel=1, dessin =1)
print(X)
