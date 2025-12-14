# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:02:18 2024

@author: frederic.ros
"""
import sys
# adding Folder_2 to the system path
#sys.path.insert(0, "..\SoftConcurrent")
sys.path.insert(0, "concurrents/ADBSCAN")
sys.path.insert(0, "concurrents/DCP_CE")
sys.path.insert(0, "concurrents/dpca")
sys.path.insert(0,"concurrents/snn-clustering-master/SNN")
sys.path.insert(0,"concurrents/RNNDBSCAN-rr")
sys.path.insert(0,"concurrents/cutESC-master/cutESC-master/code")
sys.path.insert(0,"concurrents/XMEANS_master")
sys.path.insert(0,"concurrents/KMEANS")
sys.path.insert(0,"concurrents/ANDCLUST")
sys.path.insert(0,"concurrents/PCCC")
sys.path.insert(0,"concurrents/POCS")
sys.path.insert(0,"concurrents/PIDEMUNE")
sys.path.insert(0,"concurrents/PARADIGM")
sys.path.insert(0,"concurrents/DECSIMPLIFIE")
sys.path.insert(0,"concurrents/VADE")
sys.path.insert(0,"ToolsCode")

from F_visu import plot_density_with_projection
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import spectral_clustering
from sklearn import cluster, datasets
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import spectral_clustering
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import numpy as np
from F_modelenewidee import ProximityPredictor
from F_newidee import compute_onlyinput_patterns_with_embedding
from F_mutualScan import cluster_mutual_density
from demuneCluster import DenMune, transformer_labels
from sdbscan import SDBSCAN
from DPC_CE_ALGO import dpc_ce
from cluster import DensityPeakCluster
from DECsimplified import deep_embedding_clustering
from snn import SNN
from VADE import Vade
from DataSet import GetDataSet
from RNNDBSCAN import  RNN_DBSCAN, createDistanceMatrix, buildDataSet
from Demo_RnnDbscan import rnndbscan
from collections import namedtuple
from  cutESC import  cutESC
from KmeansClustering import launchKmeans
from XmeansClustering import XMeans, XMeansTarget
from sklearn.cluster import MeanShift
from F_noise import GetListeNoise, getNovellistwithoutnoise, getDatawithout
from F_util import ClassificationScore, Selectione
from ANDClust import ANDClust
from pocs_clusering_utils import pocs_clustering
from collections import Counter
import time
try:
    import umap
    has_umap = True
except ImportError:
    has_umap = False


#******


def visualize_clusters_2d_tsne(data, labels, method='tsne',name=None):
    """
    Visualise les clusters dans un espace 2D à partir des données et des labels de clusters.
    Utilise TSNE ou UMAP si les données ont plus de 2 dimensions.

    Args:
        data (ndarray): Matrice de données (n, d) où chaque ligne représente un point.
        labels (list or ndarray): Liste des labels de clusters pour chaque point.
        method (str): 'tsne' (par défaut) ou 'umap' pour choisir la méthode de réduction si d > 2.
    """
    if data.shape[1] > 2:
        if method == 'umap':
            if not has_umap:
                raise ImportError("UMAP n'est pas installé. Installez-le avec `pip install umap-learn`.")
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            reducer = TSNE(n_components=2, random_state=42)
        data_2d = reducer.fit_transform(data)
    else:
        data_2d = data

    # Couleurs des clusters
    unique_labels = set(labels)
    cluster_colors = {label: plt.cm.jet(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 8))
    '''
    for label in unique_labels:
        cluster_points = data_2d[np.array(labels) == label]
        if label == -1:
            j=1
            #plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color='gray', label='Noise')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_colors[label], label=f'Cluster {label}')
    '''
  # --- Étape 1 : tracer le bruit d'abord ---
    if -1 in unique_labels:
        noise_points = data_2d[np.array(labels) == -1]
        '''
        plt.scatter(noise_points[:, 0], noise_points[:, 1],
                    color='lightgray', alpha=0.4, s=20, label='Noise')
        '''
        plt.scatter(noise_points[:, 0], noise_points[:, 1],
                    color='black', alpha=0.4, s=20, label='Noise')
    # --- Étape 2 : tracer les clusters ---
    for label in unique_labels:
        if label == -1:
            continue  # déjà tracé
        cluster_points = data_2d[np.array(labels) == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=cluster_colors[label],  label=f'Cluster {label}')
    if name == None: 
        plt.title("2D Visualization of Clusters"+" (D-space="+str(data.shape[1])+")")
    else: 
        plt.title(name)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
#    plt.legend()
    plt.show()

#**********************************************************************************
def DrawConcurrent(data, y,title):
    #print(matplotlib.get_backend())
    visualize_clusters_2d_tsne(data, y, method='tsne',name=title)
    
#**********************************************************************************
def filtresilhouette(data, label):
    
    LabelUnique = np.unique(label) #le nombre de labels unique...
    ListL = list(label)
    labelnovel= label.copy()
    C = 0
    x=[]
    for i in range(0,len(LabelUnique)):
        t = ListL.count(LabelUnique[i])
        if t == 1:  # 1 seul...
           x.append(LabelUnique[i])     
        else:
            C=C+1
    if C >= 2:
        for i in range(0,len(label)):
            for j in range(0,len(x)):
                if labelnovel[i] == x[j]:
                    labelnovel[i] = 1000
        return 1, labelnovel
    else:
        return 0, labelnovel        
#**********************************************************************************
def getSilhouettebase(data, y_true,noise=0, label_noise=-1):
    y_true_without = y_true.copy()
    data_out = data.copy()

    if noise != 0:
        L_noise = GetListeNoise(y_true, label_noise)
        y_true_without = getNovellistwithoutnoise(y_true,L_noise)
        data_out = getDatawithout(data, L_noise)
        
    S =silhouette_score(data_out, y_true_without)
    return S
    
def getIndexes(data, y_true, y, noise, noise_m, label_noise,modeselection=0):   
    
    #label_init_noise = 0
    label_init_noise = label_noise
    y_true_without = y_true.copy()
    y_without = y.copy()
    
    data_out = data.copy()
    
    if noise != 0:
        L_noise = GetListeNoise(y_true, label_init_noise)
        y_true_without = getNovellistwithoutnoise(y_true,L_noise)
        y_without = getNovellistwithoutnoise(y,L_noise)
        data_out = getDatawithout(data, L_noise)
        #print("taille noise", len(L_noise), len(y_true_without), len(y_without))
        
    
#    Cp = (int)(len(np.unique(y_without))-noise_m)
    Cp = (int)(len(np.unique(y))-noise_m)
    S = 0
    if modeselection == 1:
        if (Cp > 1) and (Cp<20): S =silhouette_score(data_out, y_without)
        return 0,0,S,0,Cp
    
    y = np.array(y)
    y_without = np.array(y_without)
    y_true_without = np.array(y_true_without)
    
    Z1 = Z2 = 0
    if noise_m == 1:
        #print("Shape", y_true_without.shape, y_without.shape)
        #print("label noise",label_noise,"y",y.shape)
        I = y_without.copy()
        Z1 = y_true_without[I!=label_noise]
        Z2 = y_without[I!=label_noise]
        #print("Shape", Z1.shape, Z2.shape)
        M = adjusted_mutual_info_score(Z1, Z2, average_method='arithmetic')
        R = adjusted_rand_score(Z1, Z2)
    else:
        M = adjusted_mutual_info_score(y_true_without, y_without, average_method='arithmetic')
        R = adjusted_rand_score(y_true_without, y_without)
 
    #np.savetxt("essaiAvant.txt", y_without)
   
    Ct = (int)(len(np.unique(y_true))-noise)
    #Cp = (int)(len(np.unique(y_without))-noise_m)
    Cp = (int)(len(np.unique(y))-noise_m)
    #np.savetxt("label.txt", y_without)
    if Cp>1:
        if noise_m==1: #il y a du bruit...
            data_test = data_out[I!=label_noise]
            label_test = Z2
            #print("SHAPE",data_test.shape, label_test.shape)
            G, label_test = filtresilhouette(data_test, label_test)
            if G==1: S =silhouette_score(data_test, label_test)
        else: 
            G, label_test = filtresilhouette(data_out, y_without)
            if G == 1: S =silhouette_score(data_out, label_test)
 #   print("(M,R,S):",M,R,S,"C_true",Ct, "C_pred", Cp)
    
    return M,R,S,Ct,Cp

def G_statistics(Y_result):
    n = len(Y_result)
    Y_result = np.array(Y_result)
    print("shape",Y_result.shape)
    n = Y_result.shape[0]
    p = Y_result.shape[1] #les méthodes
    q = Y_result.shape[2] #les critères
    R = np.zeros((p,6,2),float)
    for m in range(0,p):
        for i in range(0,6):
            if i==3 or i==4: continue 
            V = Y_result[:,m,i]
            R[m][i][0] = np.mean((V))
            R[m][i][1] = np.std(V)
            if i==5: 
                R[m][i][0] = np.mean(V[V!=0])   
                R[m][i][1] = 0
                
        C1 = Y_result[:,m,3]
        C2 = Y_result[:,m,4]
        
        VC = np.zeros(len(C1), float)
        Nb_zero = 0
        for u in range(0,len(C1)):
            Smin = min(C1[u], C2[u])
            Smax = max(C1[u], C2[u])
            if Smax != 0:
                VC[u] = Smin / Smax
                Nb_zero = Nb_zero + (Smax - Smin == 0)
            else:
                VC[u] = 0
        
        R[m][4][0] = np.mean((VC[VC!=0]))
        R[m][4][1] = np.std(VC[VC!=0])
        R[m][3][0] = Nb_zero / n # le pourcentage de bonne classification exacte.
        R[m][3][1] = 0
 
    return R
#*****************************************************************************************
def R_Kmeans(data, y_true, noise, noise_methode,draw=0):

        start = time.time() 
        labels = launchKmeans(data, 2, 9)
        end = time.time()
        duration = (end - start)/6
        y = labels
        if draw==1:
            DrawConcurrent(data, labels,"KMEANS")
                
        
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"KMEANS(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    
def R_Xmeans  (data, y_true, noise, noise_methode,draw=0):
    start = time.time()
    xm = XMeans(data, kmax=20)
    #xm = XMeansTarget(data, target_k = len(np.unique(y_true))-noise, kmax=20)
    
    xm.fit(); y = xm.labels
    end = time.time()
    duration = (end - start)/1
    if draw==1:
            DrawConcurrent(data, xm.labels,"X MEANS")
        
    M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
    CC = ClassificationScore(data, y,noise_methode,5)
    print(f"XMEAN(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
    if (Cp==1):
        print("no discovery XMEANS")
        return 0,0,0,Ct,0,0
    else:
        return M,R,S,Ct,Cp,CC

def R_MeanShift(data, y_true, noise, noise_methode,draw=0):
    
    duration = 0
    G = False
    rangk = [None, 2, 5, 10]; S_max = -1; k_max=0; 
    for k in rangk:    
        clustering = MeanShift(bandwidth=k).fit(data)
        nb_clusters = len(np.unique(clustering.labels_))
        if nb_clusters > 1:
            M,R,S,Ct,Cp = getIndexes(data, y_true, clustering.labels_, noise, noise_methode, label_noise=-1,modeselection=1)
            if S > S_max: S_max = S; k_max=k
    
    if (S_max !=-1):
        start = time.time()
        clustering = MeanShift(bandwidth=k_max).fit(data)
        end = time.time()
        duration = (end - start)/1
        y = clustering.labels_
        nb_classe_ok = len(np.unique(y)) > 1
        if nb_classe_ok == True:
            M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
            y = clustering.labels_
            CC = ClassificationScore(data, y,noise_methode,5)
            label_counts = Counter(y)
            result = [(label, count) for label, count in label_counts.items()]
            #print(result)
            G = Selectione(result,0.8, 50)
    
            print(f"MEANSHIFT(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
                  f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
            if draw==1:
                DrawConcurrent(data, y,"MEANSHIFT")
                
    if G == True:
        return M,R,S,Ct,Cp,CC
    else:
        return 0,0,0,Ct,0,0
    
def R_DBSCAN(data, y_true, noise, noise_methode,draw=0):            
    lnoise = -1
    n = data.shape[0]
    
    rangk = [0.1, 0.2, 0.2, 0.4, 0.5]
    rangl = [(int)(0.1*np.sqrt(n)), (int)(0.2*np.sqrt(n)),(int)(0.3*np.sqrt(n)),
             (int)(0.4*np.sqrt(n)),(int)(0.5*np.sqrt(n))]
    S_max = -1; k_max=0; l_max =0; Cp_max=0
    for k in rangk:
        for l in rangl:
            z = DBSCAN(eps=k, min_samples=l).fit(data)
            y = z.labels_
            noise_methode = (int)(len(y[y==lnoise]) !=0) #s'il y a du bruit...
            M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=lnoise,modeselection=1)
            label_counts = Counter(y)
            result = [(label, count) for label, count in label_counts.items()]
            #print(result)
            G = Selectione(result,0.8, 50)
    
            if S > S_max and (Cp>=2 and Cp<=20 and G==True): S_max = S; k_max=k; l_max = l; Cp_max = Cp
    
    #print("...",S_max,k_max,l_max, Cp_max)
    if S_max != -1:
        #print("CPMAX", Cp_max)
        start = time.time()
        z = DBSCAN(eps=k_max, min_samples=l_max).fit(data)
        end = time.time()
        duration = (end-start)/1
        y = z.labels_
        if draw==1:
            DrawConcurrent(data, y,"DBSCAN")
        noise_methode = (int)(len(y[y==lnoise]) !=0) #s'il y a du bruit...
       
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode, label_noise=lnoise,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"DBSCAN(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        #np.savetxt("dbscan.txt", y)
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery DBSCAN")
        return 0,0,0,Ct,0,0

def R_ADBSCAN(data, y_true, noise, noise_methode,draw=0):
    
    n = data.shape[0]
    '''
    rangk = [(int)(0.1*np.sqrt(n)), (int)(0.2*np.sqrt(n)),(int)(0.3*np.sqrt(n)),
             (int)(0.4*np.sqrt(n)),(int)(0.5*np.sqrt(n))]
    '''
    rangk = [(int)(0.9*np.sqrt(n)), (int)(0.95*np.sqrt(n)),(int)(1*np.sqrt(n)),
             (int)(1.1*np.sqrt(n)),(int)(1.2*np.sqrt(n))]
    
    #rangk = [(int)(1*np.sqrt(n))]
             
    #rangl = [0.01, 0.02, 0.03, 0.04, 0.05]
    rangl = [0.05, 0.06, 0.07, 0.08, 0.09]
    #rangl = [0.05]
    
    S_max = -1; k_max=0; l_max = 0 
    for k in rangk:
        for l in rangl:
            y = SDBSCAN(min_samples=k,noise_percent=l).fit_predict(data)
            M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-1,modeselection=1)
            #print(k,l,S)
            if S > S_max and (Cp>=2 and Cp<=20): S_max = S; k_max=k; l_max = l
        
    if S_max != -1:
        start = time.time()
        y = SDBSCAN(min_samples=k_max,noise_percent=l_max).fit_predict(data)   
        end = time.time()
        duration = (end-start)/1    
        if draw==1:
            DrawConcurrent(data, y,"ADBSCAN")
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"ADBSCAN(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery ADBSCAN")
        return 0,0,0,Ct,0,0
        
def R_DPCA(data, y_true, noise, noise_methode,draw=0):
    #print("DPCA") # plot decision graph to set params `density_threshold`, `distance_threshold`.
    n = data.shape[0]
    
    rangk = [(int)(0.1*np.sqrt(n)), (int)(0.2*np.sqrt(n)),(int)(0.3*np.sqrt(n)),
             (int)(0.4*np.sqrt(n)),(int)(0.5*np.sqrt(n))]
    rangl = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    S_max = -1; k_max=0; l_max = 0 
    for k in rangk:
        for l in rangl:
            dpca = DensityPeakCluster(density_threshold=k, distance_threshold=l, anormal=False)
            dpca.fit(data); y = dpca.labels_
            M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-1,modeselection=1)
            #print(k,l,S)
            if S > S_max and (Cp>=2 and Cp<=20): S_max = S; k_max=k; l_max = l
    
    if S_max != -1:    
        start = time.time()
        dpca = DensityPeakCluster(density_threshold=k_max, distance_threshold=l_max, anormal=False)
        end = time.time()
        duration = (end-start)/1
        dpca.fit(data); y = dpca.labels_
        if draw==1:
            DrawConcurrent(data, y,"DPCA")
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
        y = dpca.labels_
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"DPCA(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC

    print("no discovery DPCA")
#    np.savetxt("dpca.txt", y)
    return 0,0,0,Ct,0,0

def R_DPCCE(data, y_true, noise, noise_methode,draw=0):
#    print("DPC-CE") # plot decision graph to set params `density_threshold`, `distance_threshold`.
   
    
    rangk = [0.1, 0.2, 0.3, 0.4, 0.5]
    rangl = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    S_max = -1; k_max=0; l_max = 0 
    for k in rangk:
        for l in rangl:
            y = dpc_ce(data, rhomin = k, deltamin = l)
            M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-1,modeselection=1)
            #print(k,l,S)
            if S > S_max and (Cp>=2 and Cp<=20): S_max = S; k_max=k; l_max = l
    if S_max != -1:    
        start = time.time()
        y = dpc_ce(data, rhomin = k_max, deltamin = l_max)
        end = time.time()
        duration = (end - start)/1
        if draw==1:
            DrawConcurrent(data, y,"DPC-CE")
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"DPCE(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC

    print("no discovery DPCE")
    return 0,0,0,Ct,0,0

def R_HDBSCAN(data, y_true, noise, noise_methode,draw=0):
    
    
    n = data.shape[0]
    rangk = [(int)(0.01*n), (int)(0.02*n), (int)(0.03*n), (int)(0.04*n), (int)(0.05*n)]; S_max = -1; k_max=0; 
    
    S_max = -1; k_max=0
    for k in rangk:
        hdb = HDBSCAN(min_cluster_size=k)
        hdb.fit(data); y = hdb.labels_
        noise_methode = (int)(len(y[y==-1]) !=0) #s'il y a du bruit...
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_m = noise_methode, label_noise=-1,modeselection=1)
        label_counts = Counter(y)
        result = [(label, count) for label, count in label_counts.items()]
            #print(result)
        G = Selectione(result,0.8, 50)

        if S > S_max and Cp >=2 and G==True: S_max = S; k_max=k
        
    
    if k_max != 0:
        start = time.time()
        hdb = HDBSCAN(min_cluster_size=k_max)
        hdb.fit(data); y = hdb.labels_
        end = time.time()
        duration = (end - start)/1
        if draw==1:
            DrawConcurrent(data, y,"HDBSCAN")
        noise_methode = (int)(len(y[y==-1]) !=0) #s'il y a du bruit...
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_m=noise_methode, label_noise=-1,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5) 
        print(f"HDBSCAN(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery HDBSCAN")
        return 0,0,0,Ct,Cp,0

def R_SNN(data, y_true, noise, noise_methode,draw=0):
    #print("SNN") # plot decision graph to set params `density_threshold`, `distance_threshold`.
    n = data.shape[0]
    rangk = [(int)(0.1*np.sqrt(n)), (int)(0.2*np.sqrt(n)),(int)(0.3*np.sqrt(n)),
             (int)(0.4*np.sqrt(n)),(int)(0.5*np.sqrt(n))]
    
    rangl = [0.1, 0.2, 0.3, 0.4, 0.5]
    #rangl = [0,3,0.4, 0.5]
    
    S_max = -1; k_max=0; l_max = 0 
    for k in rangk:
        for l in rangl:
            snn = SNN(neighbor_num=k, min_shared_neighbor_proportion=l)
            z = snn.fit(data); y = z.labels_
            M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-1,modeselection=1)
            #print(k,l,S)
            if S > S_max and (Cp >=2 and Cp<=20): S_max = S; k_max=k; l_max = l
    if S_max!=-1:    
        start = time.time()
        snn = SNN(neighbor_num=k_max, min_shared_neighbor_proportion=l_max)
        z = snn.fit(data); y = z.labels_
        end = time.time()
        duration = (end - start)/1
        if draw==1:
            DrawConcurrent(data, y,"SNN")
            
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"SNN(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery SNN")
        return 0,0,0,Ct,0,0
    
def R_SPECTRAL(data, y_true, noise, noise_methode,draw=0):
    #print("SPECTRAL")
    affinity = pairwise_kernels(data, metric='rbf')
    rangk = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15]; S_max = -1; k_max=0; 
    
    S_max = - 1    
    for k in rangk:
        y = spectral_clustering(affinity=affinity, n_clusters=k, assign_labels="discretize", 
                             random_state=0)
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-1,modeselection=1)
        label_counts = Counter(y)
        result = [(label, count) for label, count in label_counts.items()]
            #print(result)
        G = Selectione(result,0.8, 50)
        
        if (S > S_max and G==True): S_max = S; k_max=k
        
    if S_max != -1:
        start = time.time()
        y = spectral_clustering(affinity=affinity, n_clusters=k_max, assign_labels="discretize", 
                                random_state=0)
        end = time.time()
        duration = (end - start)/1
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-1,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"SPECTRAL(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        if draw==1:
            DrawConcurrent(data, y,"SPECTRAL")
    
        return M,R,S,Ct,Cp,CC
    else:
        return 0,0,S,0,0,0
    


def R_RNNDBSCAN(data, y_true, noise, noise_methode,draw=0):
    #print("RNN DBSCAN")
  
    n = data.shape[0]
    rangk = [(int)(0.1*np.sqrt(n)), (int)(0.2*np.sqrt(n)),(int)(0.3*np.sqrt(n)),
             (int)(0.4*np.sqrt(n)),(int)(0.5*np.sqrt(n))]
    S_max = -1; k_max=0;
    
    for k in rangk:
         y = rnndbscan(data, y_true,k)
         M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-2,modeselection=1)
         label_counts = Counter(y)
         result = [(label, count) for label, count in label_counts.items()]
         #print(result)
         G = Selectione(result,0.8, 50)
    
         if S > S_max and (Cp>=2 and Cp<=20) and G==True: S_max = S; k_max=k
    
    if S_max != -1:
        start = time.time()
        y = rnndbscan(data, y_true,k_max)
        end = time.time()
        duration = (end - start)/1
        if draw==1:
            DrawConcurrent(data, y,"RNN DBSCAN")
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-2,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"RNNDBSCAN(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery RNNDBSCAN")
        return 0,0,0,Ct,0,0

def R_ANDCLUST(data, y_true, noise, noise_methode,draw=0):
    #print("RNN DBSCAN")
    N = 30
    eps = 0.2
    k = 9
    b_width = 4
    krnl = 'tophat'
    
    n = data.shape[0]
    rangk = [(int)(0.1*np.sqrt(n)), (int)(0.2*np.sqrt(n)),(int)(0.3*np.sqrt(n)),
             (int)(0.4*np.sqrt(n)),(int)(0.5*np.sqrt(n))]
    rangl = [0.1,0.3,0.5,0.7,1]
    
    S_max = -1; k_max=0;
    #print("noise",noise,"noise methode",noise_methode)
    for k in rangk:
        for eps in rangl:
            andClust=ANDClust(data,N,k,eps,krnl,b_width)
            y=andClust.labels_ 
            M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-2,modeselection=1)
            label_counts = Counter(y)
            result = [(label, count) for label, count in label_counts.items()]
            #print(result)
            G = Selectione(result,0.8, 50)
            if S > S_max and (Cp>=2 and Cp<=20 and G==True): S_max = S; k_max=k; e_max = eps
    
    if S_max != -1:
        start = time.time()
        andClust=ANDClust(data,N,k_max,e_max,krnl,b_width)
        y=andClust.labels_
        end = time.time()
        label_counts = Counter(y)
        result = [(label, count) for label, count in label_counts.items()]
        #print("SELEC",G)
        #print(result)
        #print("ANDCLUST",'time',(end - start)/1)
        duration = (end - start)/1
        if draw==1:
            DrawConcurrent(data, y,"ANDCLUST")
            
            
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-2,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"ANDCLUST(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery ANDCLUST")
        return 0,0,0,Ct,0,0
   
def R_POCS(data, y_true, noise, noise_methode,draw=0):
   
    n = data.shape[0]
    
    rangk = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15]
    
    S_max = -1; k_max=0;
    
    for num_clusters in rangk:
        centroid_list_pocs, label_list_pocs, proctime_list_pocs = pocs_clustering(data, num_clusters, 100)
        #centroid_list_pocs, label_list_pocs, proctime_list_pocs = pocs_clustering(data, k, 100)
        y = label_list_pocs[-1]
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise, noise_methode, label_noise=-2,modeselection=1)
        if S > S_max and (Cp>=2 and Cp<=20): S_max = S; k_max=num_clusters;
    
    if S_max != -1:
        start = time.time()
        centroid_list_pocs, label_list_pocs, proctime_list_pocs = pocs_clustering(data, k_max, 100)
        y = label_list_pocs[-1]
        end = time.time()
        duration = (end - start)/1       
               
        # Comptage des labels
        label_counts = Counter(y)
        # Affichage du résultat sous forme de liste
        result = [(label, count) for label, count in label_counts.items()]
        #print("Nombre d'individus par label :", result)
        if draw==1:
            DrawConcurrent(data, y,"POCS")
            
        M,R,S,Ct,Cp = getIndexes(data, y_true, y, noise,noise_methode,label_noise=-2,modeselection=0)
        CC = ClassificationScore(data, y,noise_methode,5)
        print(f"POCS(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery POCS")
        return 0,0,0,Ct,0,0



def R_DEMUNE(data, y_true, noise, noise_methode,draw=0):    

    rangk = [5,10,15,20,25]
    S_max = -1; k_max=0;
    
    for knn in rangk:
        dm = DenMune(train_data=data, k_nearest=knn)
        labels, validity = dm.fit_predict(show_analyzer=False, show_noise=True)
        dict_labels, validity = dm.fit_predict(show_analyzer=False, show_noise=True)
        labels = transformer_labels(dict_labels)
        M,R,S,Ct,Cp = getIndexes(data, y_true, labels, noise, noise_methode, label_noise=-1,modeselection=1)
        #print("Syl",S)
        if S > S_max and (Cp>=2 and Cp<=20): S_max = S; k_max=knn;
    
    if S_max != -1:
        start = time.time()
        dm = DenMune(train_data=data, k_nearest=k_max)
        labels, validity = dm.fit_predict(show_analyzer=False, show_noise=True)
        dict_labels, validity = dm.fit_predict(show_analyzer=False, show_noise=True)
        labels = transformer_labels(dict_labels)
        end = time.time()
        if draw==1:
            DrawConcurrent(data, labels,"DENMUNE")
        M,R,S,Ct,Cp = getIndexes(data, y_true, labels, noise,noise_methode,label_noise=-1
                                 ,modeselection=0)
        CC = ClassificationScore(data, labels,noise_methode,5)
        duration = end - start
        print(f"DEMINE(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery DEMUNE")
        return 0,0,0,Ct,0,0
    
def R_OURS(data, y_true, noise, noise_methode,p_embedding=10,namemodel="",
           ourmethod_parameter=[0.4,0.8],draw=0,normalize_distances=False):
    Kv_in = 16
    Kv_out = 16
    Kv_mutual = 16
    S_thresh = 0.8 #similarity degree
    time_1 = time_2 = 0
    display_density = False   
        
    input_dim = Kv_in * Kv_in + p_embedding 
    output_dim = Kv_out
    start = time.time()
    model = ProximityPredictor(input_dim=input_dim, output_dim=output_dim)
    
    model.load_model(namemodel)
    in_pat = compute_onlyinput_patterns_with_embedding(data, p_embedding=p_embedding, Kv_in=Kv_in, 
                                                               embedding_dict=None, random_state=None,
                                                               normalize_distances=normalize_distances)    
#    start1 = time.time()
    memberhsip = model.predict(in_pat)
#    end1 = time.time()
    end = time.time()
    
    
    
    
    
    time_1 = end - start
    
    #rangk = [0.4, 0.6, 0.8]
    rangk = ourmethod_parameter
    #rangk = [0.7,0.8]
    
    S_max = -1; d_max=-10;
    #print("parameteres",ourmethod_parameter)
    for density in rangk:       
        labels, densities, G = cluster_mutual_density(data, memberhsip, Kv=Kv_mutual, 
                                                      threshold=S_thresh, 
                                                      min_density=density, 
                                                      normalize=True)
        M,R,S,Ct,Cp = getIndexes(data, y_true, labels, noise, noise_methode, 
                                 label_noise=-1,modeselection=0)
        
        if S > S_max and (Cp>=2 and Cp<=20): 
            S_max = S; d_max=density;
    
    if display_density == True:
        plot_density_with_projection(data, densities, method="tsne", 
                                     title="Projection avec densité", cmap='Reds', random_state=42)

    if S_max != -1:
        start = time.time()
        labels, densities, G = cluster_mutual_density(data, memberhsip, Kv=Kv_mutual, 
                                                      threshold=S_thresh, 
                                                      min_density=d_max, 
                                                      normalize=True)
        end = time.time()
        M,R,S,Ct,Cp = getIndexes(data, y_true, labels, noise, noise_methode, 
                                 label_noise=-1,modeselection=0)
        if draw==1:
            DrawConcurrent(data, labels,"OURS")
        
        CC = ClassificationScore(data, labels,noise_methode,5)
        time_2 = end - start
#        print("Smaxx",end1 - start1,time_1,time_2)
        duration = time_1 + time_2
        print(f"OURS(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
              f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")
        return M,R,S,Ct,Cp,CC
    else:
        print("no discovery OURS")
        return 0,0,0,Ct,0,0
    
def R_DECSIMPLIFIE(data, y_true, noise,noise_methode, draw=0):
    latent_compression = int(0.25 * data.shape[1])
    latent_dim = max(10, latent_compression)
    n_clusters_low=2
    n_clusters_up=15
    
    start = time.time()
    Z, y_pred = deep_embedding_clustering(data, y_true, n_clusters_low=n_clusters_low, 
                                          n_clusters_up=n_clusters_up,
                                          latent_dim=latent_dim, epochs=100)
    end = time.time()
    duration = (end - start)/(n_clusters_up - n_clusters_low)
    
    if draw==1:
        DrawConcurrent(data, y_pred,"DEC")
        
    M,R,S,Ct,Cp = getIndexes(data, y_true, y_pred, noise,noise_methode,label_noise=-1,
                             modeselection=0)
    CC = ClassificationScore(data, y_pred,noise_methode,5)
    print(f"DEC(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
      f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")

    return M,R,S,Ct,Cp,CC

def R_VADE(data, y_true, noise,noise_methode, draw=0):
    latent_compression = int(0.25 * data.shape[1])
    latent_dim = max(10, latent_compression)
    start = time.time()
    n_clusters_low=2
    n_clusters_up=15
    
    Z, y_pred = Vade(data, y_true, n_clusters_low=n_clusters_low, 
                     n_clusters_up=n_clusters_up, latent_dim=latent_dim, epochs=100)
    
    end = time.time()
    duration = (end - start)/(n_clusters_up - n_clusters_low)
    if draw==1:
        DrawConcurrent(data, y_pred,"VADE")
    M,R,S,Ct,Cp = getIndexes(data, y_true, y_pred, noise,noise_methode,label_noise=-1,modeselection=0)
    CC = ClassificationScore(data, y_pred,noise_methode,5)
    print(f"VADE(time,M,R,S): {duration:.3f}, {M:.3f}, {R:.3f}, {S:.3f} | "
      f"C_true: {Ct:.3f}, C_pred: {Cp:.3f}, Cl: {CC:.3f}")

    return M,R,S,Ct,Cp,CC

def getnoisemethode(noise_methode):
    noise_methode[0] = 0
    noise_methode[1] = 0
    noise_methode[2] = 0
    noise_methode[3] = 1
    noise_methode[4] = 1 
    noise_methode[5] = 1
    noise_methode[6] = 1
    noise_methode[7] = 1
    noise_methode[8] = 1
    noise_methode[9] = 0
    noise_methode[10] = 1
    noise_methode[11] = 1
    noise_methode[12] = 1
    noise_methode[13] = 1
    noise_methode[14] = 1
    noise_methode[15] = 0
    noise_methode[16] = 0
    
#**********************************************************************************
def Testcompetitors(data, y_true, run, noise, noise_methode,  
                    namemodel="models/model"+"hybrid2-10"+".pth",
                    ourmethod_parameter =[0.4,0.8],draw=True):
   
    #namemodel="models/model"+"hybrid2-30"+".pth"
    X = []
    #KMEANS
    if (run[0]==True): X.append(R_Kmeans(data, y_true, noise, noise_methode[0], draw))
          
    #XMEANS
    if (run[1]==True): X.append( R_Xmeans(data, y_true, noise, noise_methode[1], draw))
        
    #MEANSHIFT
    if (run[2]==True): X.append  (R_MeanShift(data, y_true, noise, noise_methode[2], draw))
                
    #DBSCAN  
    if (run[3]==True): X.append(R_DBSCAN(data, y_true, noise, noise_methode[3], draw))
            
    # ADBSCAN
    if (run[4]==True): X.append(R_ADBSCAN(data, y_true, noise, noise_methode[4], draw))
        
    # dpca model
    if (run[5]==True): X.append(R_DPCA(data, y_true, noise, noise_methode[5], draw))
        
    #dpc_ce
    if (run[6]==True): X.append(R_DPCCE(data, y_true, noise, noise_methode[6], draw))
            
    #HDBSCAN
    if (run[7]==True): X.append(R_HDBSCAN(data, y_true, noise, noise_methode[7], draw))
        
    #SNN
    if (run[8]==True): X.append(R_SNN(data, y_true, noise, noise_methode[8], draw))
        
    #SPECTRAL
    if (run[9]==True): X.append(R_SPECTRAL(data, y_true, noise, noise_methode[9], draw))
        
    #print("Spectral",labels)

    if (run[10]==True): X.append(R_RNNDBSCAN(data, y_true, noise, noise_methode[10], draw))
            
    if (run[11]==True): X.append(R_ANDCLUST(data, y_true, noise, noise_methode[11], draw))
        
    if (run[12]==True): X.append(R_POCS(data, y_true, noise, noise_methode[12], draw))
    
    if (run[13]==True): X.append(R_DEMUNE(data, y_true, noise, noise_methode[13], draw))
    
    if (run[14]==True): X.append(R_OURS(data, y_true, noise, noise_methode[14],p_embedding=10, 
                                        namemodel=namemodel,ourmethod_parameter=ourmethod_parameter,draw=draw))
    
    if (run[15]==True): X.append(R_DECSIMPLIFIE(data, y_true, noise,noise_methode[15], draw=draw))
    
    if (run[16]==True): X.append(R_VADE(data, y_true, noise,noise_methode[16], draw=draw))
 
    return X

#Testcompetitors(data, y_true)