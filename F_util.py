# -*- coding: utf-8 -*-
"""
Created on Fri May  9 06:40:41 2025

@author: frederic.ros
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


import numpy as np


def processstat(R,block_size=5):
   #print("SHAPE = ",np.array(R))   
    data_2d = np.array(R).squeeze()
    #print(data_2d)
    block_size = block_size
    # s'assurer que le nombre de lignes est multiple de block_size
    n_blocks = data_2d.shape[0] // block_size
    # calcul des moyennes par bloc
    means = data_2d[:n_blocks * block_size].reshape(n_blocks, block_size, -1).mean(axis=1)
    # affichage
    return means

def processstat_global(R, tri=True, CVI=None, threshold=0.1):
    """
    Calcule les moyennes et écarts-types colonne par colonne d'une matrice R,
    en filtrant éventuellement les lignes selon un CVI et un seuil.

    Parameters
    ----------
    R : array-like
        Données à traiter (N, D).
    tri : bool, optional
        Si True, ne garde que les lignes dont CVI > threshold.
    CVI : list or np.ndarray, optional
        Tableau 1D de même longueur que le nombre de lignes de R.
        Si None et tri=True, aucun filtrage n'est appliqué.
    threshold : float, optional
        Seuil utilisé pour filtrer les lignes selon CVI.

    Returns
    -------
    means : np.ndarray
        Moyenne par colonne.
    stds : np.ndarray
        Écart-type par colonne.
    n_used : int
        Nombre de lignes utilisées après filtrage.
    """
    data = np.array(R).squeeze()
    
    # Vérification de cohérence
    if data.ndim == 1:
        data = data[:, None]
    
    if tri and CVI is not None:
        CVI = np.array(CVI).squeeze()
        if len(CVI) != len(data):
            raise ValueError("La taille de CVI doit correspondre au nombre de lignes de R.")
        mask = CVI > threshold
        data_filtered = data[mask]
    else:
        data_filtered = data

    # Moyenne et écart-type par colonne
    means = np.mean(data_filtered, axis=0)
    stds = np.std(data_filtered, axis=0)
    
    return means, stds, len(data_filtered)

def G_Save(Y_result):
    n = len(Y_result)
    Y_result = np.array(Y_result)
    print("shape",Y_result.shape)
    n = Y_result.shape[0] #les images
    q = Y_result.shape[1] #les methodes
    m = Y_result.shape[2] #les critères
    print(n,q,m)
    
    A = np.zeros((n,m), float)
    for u in range(0,q):
        for i in range(0,n):
            for j in range(0,m):
                A[i][j] = Y_result[i][u][j]
        name =  "concurrent" + str(u) + ".txt"   
        A = np.round(A,2)
        print(A)
        np.savetxt(name, A, fmt='%.2e')
'''
def compute_custom_stats(data_3d, method_index, true_k_index=3, pred_k_index=4):
    """
    Calcule la moyenne et l'écart-type par critère pour une méthode donnée,
    avec transformation spécifique sur les critères true_k et pred_k.

    Paramètres :
        data_3d : ndarray de forme (nb_datasets, nb_methods, nb_criteres)
        method_index : int, index de la méthode sélectionnée
        true_k_index : int, index du critère 'nombre de clusters réels'
        pred_k_index : int, index du critère 'nombre de clusters prédits'

    Retour :
        means : ndarray de forme (nb_criteres,), moyennes par critère
        stds : ndarray de forme (nb_criteres,), écarts-types par critère
        extracted_2d : ndarray de forme (nb_datasets, nb_criteres), données transformées
    """
    if method_index < 0 or method_index >= data_3d.shape[1]:
        raise IndexError("L'index de la méthode est hors limites.")

    # Extraire les résultats pour la méthode sélectionnée
    extracted = data_3d[:, method_index, :]  # (nb_datasets, nb_criteres)
    transformed = extracted.copy()
    
    print("transformed", transformed)
    
    # Extraire les deux colonnes concernées
    true_k = extracted[:, true_k_index]
    pred_k = extracted[:, pred_k_index]

    # Critère 1 : égalité exacte (0 ou 1)
    equality_score = (true_k == pred_k).astype(float)

    # Critère 2 : min/max ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        minmax_ratio = np.true_divide(np.minimum(true_k, pred_k), np.maximum(true_k, pred_k))
        minmax_ratio[np.isnan(minmax_ratio)] = 0.0  # Cas où max=0

    # Remplacer les valeurs dans les colonnes concernées
    transformed[:, true_k_index] = equality_score
    transformed[:, pred_k_index] = minmax_ratio

    # Moyenne et écart-type par critère
    means = np.mean(transformed, axis=0)
    stds = np.std(transformed, axis=0)

    return means, stds, transformed
'''
import numpy as np

def compute_custom_stats(data_3d, method_index, true_k_index=3, pred_k_index=4):
    """
    Calcule la moyenne et l'écart-type par critère pour une méthode donnée,
    avec ajout du ratio de discovery et exclusion des lignes sans cluster découvert.

    Paramètres :
        data_3d : ndarray de forme (nb_datasets, nb_methods, nb_criteres)
        method_index : int, index de la méthode sélectionnée
        true_k_index : int, index du critère 'nombre de clusters réels'
        pred_k_index : int, index du critère 'nombre de clusters prédits'

    Retour :
        means : ndarray de forme (nb_criteres+1,), moyennes par critère + ratio discovery
        stds : ndarray de forme (nb_criteres,), écarts-types par critère (hors ratio)
        transformed_filtered : ndarray de forme (nb_valid_datasets, nb_criteres)
    """
    if method_index < 0 or method_index >= data_3d.shape[1]:
        raise IndexError("L'index de la méthode est hors limites.")

    # Extraire les résultats pour la méthode sélectionnée
    extracted = data_3d[:, method_index, :]  # (nb_datasets, nb_criteres)
    #print("Extracted=", extracted)
    transformed = extracted.copy()

    # Extraire les colonnes true_k et pred_k
    true_k = extracted[:, true_k_index]
    pred_k = extracted[:, pred_k_index]

    # --- Ratio de discovery ---
    ratio_discovery = np.mean(pred_k > 0)

    # --- Filtrer les lignes valides (au moins un cluster trouvé) ---
    valid_mask = pred_k > 0
    if np.sum(valid_mask) == 0:
        # Aucun cluster trouvé sur aucun dataset
        means = np.zeros(extracted.shape[1] + 1)
        stds = np.zeros(extracted.shape[1])
        means[-1] = 0.0  # ratio_discovery
        return means, stds, np.empty((0, extracted.shape[1]))

    filtered = extracted[valid_mask]

    # Recalcul sur les données filtrées
    true_k = filtered[:, true_k_index]
    pred_k = filtered[:, pred_k_index]

    # Critère 1 : égalité exacte (0 ou 1)
    equality_score = (true_k == pred_k).astype(float)

    # Critère 2 : min/max ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        minmax_ratio = np.true_divide(np.minimum(true_k, pred_k), np.maximum(true_k, pred_k))
        minmax_ratio[np.isnan(minmax_ratio)] = 0.0  # Cas où max=0

    # Remplacer les valeurs transformées
    transformed_filtered = filtered.copy()
    transformed_filtered[:, true_k_index] = equality_score
    transformed_filtered[:, pred_k_index] = minmax_ratio

    # Moyenne et écart-type par critère
    means = np.mean(transformed_filtered, axis=0)
    stds = np.std(transformed_filtered, axis=0)

    # Ajouter le ratio de discovery à la fin
    means = np.append(means, ratio_discovery)

    return means, stds, transformed_filtered


def compute_stats_by_criterion(data_3d, method_index):
    """
    Calcule la moyenne et l'écart-type par critère pour une méthode donnée.

    Paramètres :
        data_3d : ndarray de forme (nb_datasets, nb_methods, nb_criteres)
        method_index : int, index de la méthode sélectionnée

    Retour :
        means : ndarray de forme (nb_criteres,), moyennes par critère
        stds : ndarray de forme (nb_criteres,), écarts-types par critère
        extracted_2d : ndarray de forme (nb_datasets, nb_criteres), données brutes pour la méthode
    """
    if method_index < 0 or method_index >= data_3d.shape[1]:
        raise IndexError("L'index de la méthode est hors limites.")

    # Extraire les résultats pour la méthode sélectionnée
    extracted_2d = data_3d[:, method_index, :]  # (nb_datasets, nb_criteres)

    # Moyenne et écart-type par colonne (donc par critère)
    means = np.mean(extracted_2d, axis=0)
    stds = np.std(extracted_2d, axis=0)

    return means, stds, extracted_2d

def G_simplestatistics(Y_result):
    n = len(Y_result)
    Y_result = np.array(Y_result)
    print("shape",Y_result.shape)
    n = Y_result.shape[0]
    q = Y_result.shape[1] #les critères
    R = np.zeros((7,2),float)
    
    V1 = Y_result[:,6]
    for i in range(0,7):
        if i==3 or i==4: continue
        V = Y_result[:,i]
        R[i][0] = np.mean(V[V1!=0])
        R[i][1] = np.std(V[V1!=0])
        if i==5 or i==6: 
            print(V)
            R[i][0] = np.mean(V[V1!=0])
            R[i][1] = 0
    
    C1 = Y_result[:,3]
    C2 = Y_result[:,4]
    
    VC = np.zeros(len(C1), float)
    Nb_zero = 0
    for i in range(0,len(C1)):
        Smin = min(C1[i], C2[i])
        Smax = max(C1[i], C2[i])
        if Smax != 0:
            VC[i] = Smin / Smax
            Nb_zero = Nb_zero + Smax - Smin == 0
        else:
            VC[i] = 0
        
#    print(np.maximum(C1,C2), np.minimum(C1,C2))
 #   S = np.minimum(C1,C2) / np.maximum(C1,C2)
    R[4][0] = np.mean((VC[VC!=0]))
    R[4][1] = np.std(VC[VC!=0])
    #S = np.abs(C1 - C2)
    #S = list(S)
        #print(S)
    #Nb_zero = S.count(0)
        
    R[3][0] = Nb_zero / n # le pourcentage de bonne classification exacte.
    R[3][1] = 0
        
    return R

def Selectione(clusters,seuil, taille_min):
    OK = True
    # Étape 1 : Calculer le total des items
    total_items = sum(size for _, size in clusters)

    # Étape 2 : Calculer le seuil de 90 %
    threshold = seuil * total_items

    # Étape 3 : Filtrer les clusters
    filtered_clusters = []
    current_sum = 0

    # Trier les clusters par taille décroissante pour prendre les plus grands en premier
    clusters.sort(key=lambda x: x[1], reverse=True)

    for cluster, size in clusters:
        current_sum += size
        filtered_clusters.append((cluster, size))
        if current_sum >= threshold:
            break
        
    if len(filtered_clusters)==1: OK = False
    # Afficher les clusters significatifs
    #print("Clusters représentant 90% des items :")
    for cluster, size in filtered_clusters:
        #print(f"{cluster}: {size}")
        if size < taille_min: OK = False
    
    return OK



def ClassificationScore(data, label,noise, n_neighbor):
    C = 0
    nblabel = len(np.unique(list(label)))
      
    if nblabel> noise:
        knn = KNeighborsClassifier(n_neighbors=n_neighbor)
        if noise == 1:
            Z = data[label!=0] #sans les bruits.
            L = label[label!=0]
        else:
            Z = data; L=label
        if Z.shape[0] > len(label)/10:
            knn.fit(Z,L)
            C = knn.score(Z, L)
        
    return C