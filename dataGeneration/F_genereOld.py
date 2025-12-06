# -*- coding: utf-8 -*-
"""
Created on Sun May 12 08:21:37 2024

@author: frederic.ros
"""
import numpy as np
import random as Random
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
import pandas as pd

def getT(t):
    
    T = np.zeros((t,t), float)
    for i in range(0,t):
        T[i][i] = np.random.uniform(0.85, 1)
        for j in range(0,t): 
            if (i!=j): T[i][j] = np.random.uniform(-0.5, 0.5)
    return T

def Gettknnclass(X_train, y_train, n_neighbor=3, th=0.9):
#    print(y_train[0:200])
    
    X_train_f = X_train.copy()
    y_train_f = y_train.copy() 
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    knn.fit(X_train_f, y_train_f)
    
    y = knn.predict(X_train_f) #les bonnes prédictions
#
    knn.fit(X_train_f, y) #on refait un apprentissage
    S = knn.score(X_train_f,y) #on calcule le score
   # print("Score =", S)
    return y,S

#-------------------------------------------------------------------------------    
def UniformNoiseExterne(np_noise, dim = 2, t_out = 3, t_in = 2,noise_label = -1):
    
    X = np.random.uniform(-t_out, t_out, int(dim*np_noise)) #genere sur une ligne
    X_noise = np.reshape(X, (np_noise, dim)) #on transforme...
    
 
    X_noise_select=[]
    while (len(X_noise_select) < np_noise):
        X = np.random.uniform(-t_out, t_out, int(dim*np_noise)) #genere sur une ligne
        X_noise = np.reshape(X, (np_noise, dim)) #on transforme...

        
        for i in range(len(X_noise)):
            A = 0
            for j in range(dim):
                A = A + (int)((X_noise[i][j]<= (- t_in) or X_noise[i][j]>=(+ t_in))) 
                if A==1 and len(X_noise_select) < np_noise: 
                    X_noise_select.append(X_noise[i])
        
    A = np.asarray(X_noise_select) 
    
    y_noise = np.asarray([noise_label] * np_noise)
    
    return A, y_noise
#-------------------------------------------------------------------------------
def UniformNoiseInterne(np_noise, dim = 2, t_out = 3, t_in = 2,noise_label = -1):
    return UniformNoise(np_noise, dim, t_in, noise_label)   
    
#-------------------------------------------------------------------------------    

def UniformNoise(np_noise, dim = 2, t = 1, noise_label = -1):
    X = np.random.uniform(-t, t, int(dim*np_noise)) #genere sur une ligne
    X_noise = np.reshape(X, (np_noise, dim)) #on transforme...
    y_noise = np.asarray([noise_label] * np_noise)
    
    return X_noise, y_noise
       
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


import numpy as np

def sample_unique_hypercube_vertices_h(dim, n_centers, scale=1.0, min_hamming_ratio=0.1, 
                                       adaptive=True, max_attempts=10000):
    """
    Tire n_centers sommets distincts de l'hypercube {−scale, +scale}^dim.
    
    - Tous les sommets sont distincts.
    - On impose une distance de Hamming minimale relative (min_hamming_ratio × dim).
    - Si adaptive=True, la contrainte est relâchée automatiquement si trop stricte.
    
    Paramètres
    ----------
    dim : int
        Dimension de l'hypercube.
    n_centers : int
        Nombre de sommets distincts à tirer.
    scale : float
        Amplitude des coordonnées (par défaut ±1).
    min_hamming_ratio : float
        Fraction minimale de bits différents entre deux sommets.
    adaptive : bool
        Si True, réduit la contrainte de distance si elle bloque la génération.
    max_attempts : int
        Nombre maximal de tentatives avant de relâcher la contrainte (si adaptive=True).
    """
    if n_centers > 2**dim:
        raise ValueError("Impossible : n_centers > 2**dim")

    rng = np.random.default_rng()
    centers = []
    seen = set()
    min_hamming = max(1, int(np.ceil(min_hamming_ratio * dim)))

    def hamming_distance(a, b):
        return np.sum(a != b)

    attempts = 0
    while len(centers) < n_centers:
        vertex = rng.integers(0, 2, size=dim)
        tup = tuple(vertex)
        if tup in seen:
            attempts += 1
            continue

        if all(hamming_distance(vertex, c) >= min_hamming for c in centers):
            seen.add(tup)
            centers.append(vertex)
            attempts = 0  # reset après succès
        else:
            attempts += 1

        # Si on bloque trop longtemps → on relâche la contrainte
        if adaptive and attempts > max_attempts:
            min_hamming = max(1, min_hamming - 1)
            attempts = 0
            # print(f"[Adaptation] Nouvelle distance min = {min_hamming}")

    centers = np.array(centers)
    centers = scale * (2 * centers - 1)
    return centers



#-------------------------------------------------------------------------------
def trans(y):
    z = y.copy()
    for i in range(len(y)):
        if y[i]>=2:
            z[i] = 2
    return z
#-------------------------------------------------------------------------------
def Meandinterne(data, k, nsamples):
    if nsamples == 0: return 0,0
    if len(data) == 0: return 0,0
    t=[]
    for i in range(0,nsamples): 
        t.append(Random.randint(0, len(data)-1))

    nbrs = NearestNeighbors(n_neighbors=max(1,(int)(k)),algorithm='ball_tree' ).fit(data) #

    data_t=[]
    for i in range(0,nsamples):
        data_t.append(data[t[i]])
        
    distances, indices = nbrs.kneighbors(np.asarray(data_t)) #on a les distances et les indices.
   
    M = 0
    S = 0
    for i in range(0,nsamples): 
        M = M + np.mean(distances[i])
        S = S + np.std(distances[i])
    return M/nsamples, S/nsamples    

#-------------------------------------------------------------------------------
def Getdistance(KV, pattern):

    distances, indices = KV.kneighbors(pattern.reshape(1,-1)) #on a les distances et les indices.
    return np.mean(distances[0])

#-------------------------------------------------------------------------------
def ManageNoiseInterne(Z, label, n_neighbor,X_noise_interne):
    
    n_noise_interne = X_noise_interne.shape[0]
    y_noise_interne = np.zeros(n_noise_interne, int) - 2 #on initialise à du bruit...
    
    #création du classifieur
    n = Z.shape[0]
    n_neighbor = (int)(np.sqrt(n))
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    knn.fit(Z,label)
    n_centers = max(label) + 1
#    print("NCENTRES", n_centers)
    
    
    Stat=[]
    for i in range(0,n_centers):
        data = Z[label==(i)]
        min_neighbors = len(data) -1
        n_neighbor = min(min_neighbors, n_neighbor)
        Stat.append(Meandinterne(data, n_neighbor, max(n_neighbor,10)))
#    print(Stat)   
    Prediction = knn.predict_proba(X_noise_interne) #on prédit les valeurs des bruits dans les classes.
      
    
    KV=[]
    for i in range(0,n_centers):
        data_t = Z[label==i]
        if (len(data_t) > 2):
            min_neighbors = len(data_t) -1
            n_neigh = min(min_neighbors,n_neighbor)
            KV.append(NearestNeighbors(n_neighbors=max(1,n_neigh),
                                       algorithm='ball_tree' ).fit(data_t)) #

    t = 0
    for i in range(n_noise_interne):
        R = Prediction[i]
        win = np.argmax(R)
        if np.max(R) == 1: #prediction excelente
            d_s = Getdistance(KV[win],X_noise_interne[i])
            
            if d_s <= Stat[win][0]+2*Stat[win][1]: 
                y_noise_interne[i] =  win #on la met dans cette classe...
            else:
                y_noise_interne[i] = -2 #on la met dans cette classe...
                t += 1
                            
    return y_noise_interne

#-------------------------------------------------------------------------------

def sizecluster_v(n, ncluster, sizemin):
    
    t = ncluster * sizemin
    if (t > n):
        size_min = (int)(t/ncluster)
    else:
        size_min = sizemin
    
    s = np.zeros(ncluster,(int)) + size_min #initialization à sizemin
    
    reliquat = n - size_min * ncluster
    
    alea = (int)(reliquat / 2)
    t = 0
    while (t < reliquat):
        for i in range(ncluster):
            w = Random.randint(0,alea)
            if (t + w) <= reliquat:
                s[i] += w
                t = t + w
            else:
                s[i] += (reliquat - t)
                t = t + (reliquat - t)
               
    return s

def reindex_labels(y):
    y = np.array(y)
    
    # Identifie les éléments "ambigus"
    mask_ambiguous = np.isin(y, [-1, -2])
    
    # Crée une copie pour ne pas modifier y en place
    y_new = np.full_like(y, -1)  # tous les -1 et -2 seront directement mis à -1
    
    # Extrait les labels valides (hors -1 et -2)
    valid_labels = np.unique(y[~mask_ambiguous])
    
    # Crée une correspondance : label original -> label réindexé à partir de 0
    label_map = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
    
    # Applique la remapping sur les labels valides
    for old_label, new_label in label_map.items():
        y_new[y == old_label] = new_label

    return y_new
#-------------------------------------------------------------------------------
def CreateData(n_samples=256, dim=2, p_noise=0.1, noise_t=1,max_dev=0.5,
               hamming_distance=0.01,
               n_centers=4, Score_min=0.9, 
               proba_St=20, proba_An=30, Proba_all = 80,
               Draw=0):
 
    COEFEXTERNE = 0.5
    if n_centers == 0:
        max_cluster = min(8,2**dim)
    else:
        max_cluster = min(n_centers,2**dim)
    scale = 1
    
    n_noise = (int)(n_samples * p_noise)
    
    #****************************************************************
    if n_centers == 0:
        n_centers = Random.randint(2, max_cluster)
    else:
        n_centers = max_cluster
    
    
    Centers = sample_unique_hypercube_vertices_h(dim, n_centers, scale=1.0, 
                                                 min_hamming_ratio=hamming_distance)  
    
    nbpoint = (int)(n_samples - p_noise * n_samples)
    S = sizecluster_v(nbpoint, n_centers, (int)(nbpoint/10)) #nombre de points minimal par cluster...
    '''
    if (Random.randint(1,100) < proba_St): #on travaille sur les dev.
        if (Random.randint(1,100) < Proba_all):
            std = np.random.uniform(scale*0.05, scale*max_dev, size=(n_centers,dim))
        else:
            std = np.random.uniform(scale*0.05, scale*max_dev, size=(n_centers,1))
    else: #on travaile plutôt les frontières...
        std = np.random.uniform(scale*0.35, scale*max_dev, size=(n_centers,dim))
    '''    
 
    if (Random.randint(1,100) < proba_St): #on travaille sur les dev.
        if (Random.randint(1,100) < Proba_all):
            std = np.random.uniform(scale*max_dev*0.1, scale*max_dev, size=(n_centers,dim))
        else:
            std = np.random.uniform(scale*max_dev*0.1, scale*max_dev, size=(n_centers,1))
    else: #on travaile plutôt les frontières...
        std = np.random.uniform(scale*max_dev*0.9, scale*max_dev, size=(n_centers,dim))
 
    Z, y = make_blobs(n_samples = S, n_features = dim, 
                    centers = Centers, cluster_std = std,shuffle = False)
    n = Z.shape[0]
    if (Random.randint(1,100) < proba_An):
        std = np.random.uniform(scale*0.05, scale*max_dev, size=(n_centers,dim))
        T = getT(dim);  Z = np.dot(Z, T)  # Anisotropic blobs
        
    #print(y)
 #   if Draw==1: drawcluster2D3D(Z,y,  titre="data only create") #OK
    
    #****************************************************************
    #on teste la classificaiton.    
    k = int(np.sqrt(n)/2)
    k = min(k, n-1)   # <- assure que k < n_samples
    if len(np.unique(y)) >= 2:
        y, Score = Gettknnclass(Z, y, max(k,3) + 1)  
        if Score >= Score_min: 
            Valid = 1
        else: 
            Valid=0
    else:
        Valid = 1
    
    y_pur = y.copy()
    x_pur = Z.copy()
#    if Draw==1: drawcluster2D3D(Z,y,  titre="data only corrected")
      
    n_neighbor = (int)(np.sqrt(n))   
    
#***************************************************************************************
    #NOISE EXTERNE:  BRUIT = 0
    T = Z.copy()
    U = y.copy()
    
    n_noise_externe = (int)(n_noise * COEFEXTERNE)
    
    if n_noise != 0:
    #NOISE INTENRE
        n_noise_interne = n_noise - n_noise_externe
        X_noise_interne,y_noise_interne = UniformNoiseInterne(n_noise_interne, dim, 3*noise_t, 2*noise_t,-2)
 
        neighbor = k
        if k>0:
            Label_interne = ManageNoiseInterne(Z, y, neighbor, X_noise_interne)
            T = np.concatenate((T,X_noise_interne), axis=0)
            U = np.concatenate((U,Label_interne),axis = 0)
        T_withnoise = T.copy()
        U_withnoise = U.copy()
    
#        if Draw==1: drawcluster2D3D(T,U+2,  titre="bruit interne avec correction")
 
        X_noise_ext,y_noise_ext = UniformNoiseExterne(n_noise_externe, dim, 3*noise_t, 2*noise_t,-2) #BRUIT = 0
        #if Draw==1: drawcluster2D3D(X_noise_ext,y_noise_ext,  titre="noise externe")
    
    
        T = np.concatenate((T,X_noise_ext), axis=0)
        U = np.concatenate((U,y_noise_ext),axis = 0)
#        if Draw==1: drawcluster2D3D(T,U+2,  titre="Total noise")
    else:
        T_withnoise = T.copy()
        U_withnoise = U.copy()
       
    
#***************************************************************************************
    if (n_neighbor >= 1): 
        #on vérifie que l'on sépare bien les classes.
        Classifier = KNeighborsClassifier(n_neighbors=(int)(n_neighbor))
        Classifier.fit(T_withnoise,U_withnoise) #on prend les classes pures + les bruits internes... 
        Prediction = Classifier.predict_proba(T_withnoise) #on ne classe que les purs.
        #rajout l)
        NonPur = 0   
        for i in range(Z.shape[0]): #pour les purs....
            R = Prediction[i]
            if np.max(R) != 1: #probabilité de 1.
                U[i] = -1 #AMBIGU ou mal classe  => Création des frontières...
                NonPur += 1
 #   print("Nombre de non purs", NonPur)
#    if Draw==1: drawcluster2D3D(T,U+2,  titre="Bruit+Frontieres+Classes")
#    drawcluster2D3D(Z,Cat,  titre="Pur")
    
    
    #on teste la classificaiton.    
    k= (int)(np.sqrt(n))
    Classifier = KNeighborsClassifier(n_neighbors=k)
    Classifier.fit(T,U)
    Prediction = Classifier.predict_proba(T) #on ne classe que les purs.
    #print(Prediction)
    F = np.max(Prediction, axis = 1)
    #print(F[0:10])
#    if Draw==1: drawcluster2D3D(T,F>0.8,  titre="CORRECT CLASSIFICATION")
    
    #avant....
    #U = U + 2

    U = reindex_labels(U)
    return T,U, x_pur, y_pur, Valid, F

#-------------------------------------------------------------------------------
def savedata(f_data, f_label, size, ncluster, dim, max_dev,p_noise,Score_min,n):
    
    '''
    data, y, x_pur,ypur,valid, Pred = CreateData(size, dim, p_noise, 1,max_dev, ncluster, Score_min,1)
    print("VALID = ",valid)
    drawcluster2D3D(x_pur,ypur, titre="data only corrected")
    drawcluster2D3D(data,y, titre="data created")
    '''
          
    i= 0
    while i < n:
        if dim==-1: 
            dimact=np.randint(2,10)
        else:
            dimact = dim
        data, y, x_pur,ypur,valid, Pred = CreateData(size, dimact, p_noise, 1,max_dev, ncluster, Score_min,0)
        if valid == 1:
         #   drawcluster2D3D(x_pur,ypur, titre="data only")
            drawcluster2D3D(data,y, titre="data created")
            df = pd.DataFrame(data)
            data_f = f_data + "/data"+str(i)+ ".csv"
            df.to_csv(data_f, header=None,index=False)
            df = pd.DataFrame(y)
            label_f = f_label + "/label"+str(i)+ ".csv"
            df.to_csv(label_f, header=None,index=False)
            print("data ",i,"name",data_f, label_f)   
            i = i + 1
    
#-------------------------------------------------------------------------------
