# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:05:24 2025

@author: frederic.ros
"""

import numpy as np
from collections import Counter
from sklearn.utils import shuffle

from collections import Counter
from sklearn.utils import shuffle
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def resample_to_n(X, y, n,min_class_size=10, random_state=42, augmentation_std=0.01):
    """
    Ramène le dataset à 1000 exemples en respectant les proportions des classes.
    Supprime les classes avec trop peu d'exemples (< min_class_size).
    
    Paramètres :
        X : ndarray (n_samples, n_features)
        y : ndarray (n_samples,)
        min_class_size : taille minimale d'une classe pour être conservée
        random_state : graine aléatoire
        augmentation_std : écart-type du bruit ajouté pour la data augmentation

    Retour :
        Xf, yf : nouvelles données avec 1000 éléments
    """
    np.random.seed(random_state)
    
    # Étape 1 : filtrer les classes trop petites
    class_counts = Counter(y)
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_class_size]

    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    # Étape 2 : calcul des nouvelles proportions
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    per_class_target = np.round(proportions * n).astype(int)

    # Correction pour avoir exactement 1000 items
    diff = n - per_class_target.sum()
    for i in range(abs(diff)):
        per_class_target[i % len(per_class_target)] += np.sign(diff)

    # Étape 3 : traitement classe par classe
    Xf_list, yf_list = [], []

    for cls, target_count in zip(classes, per_class_target):
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        if len(X_cls) >= target_count:
            indices = np.random.choice(len(X_cls), target_count, replace=False)
            Xf_list.append(X_cls[indices])
            yf_list.append(y_cls[indices])
        else:
            n_needed = target_count - len(X_cls)
            X_aug = X_cls[np.random.choice(len(X_cls), n_needed, replace=True)]
            noise = np.random.normal(0, augmentation_std, size=X_aug.shape)
            X_aug += noise
            Xf_list.append(np.vstack([X_cls, X_aug]))
            yf_list.append(np.hstack([y_cls, np.full(n_needed, cls)]))

    Xf = np.vstack(Xf_list)
    yf = np.concatenate(yf_list)
    return shuffle(Xf, yf, random_state=random_state)

def getPCA(X):
    if X.shape[1] > 2:
        pca = PCA(n_components=0.95)
        pca.fit(X)
        X = pca.fit_transform(X)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        return X
def load_classification_data_from_directory(directory_path,n=1000):
    """
    Charge les données et les labels de tous les fichiers .txt d'un répertoire donné.
    
    Parameters:
    - directory_path (str): Chemin du répertoire contenant les fichiers .txt.
    
    Returns:
    - data_list (list of DataFrame): Liste des données (features) de chaque fichier.
    - labels_list (list of Series): Liste des labels (dernière colonne) de chaque fichier.
    """
    data_list = []
    labels_list = []
    files = []
    sizes = []
    p = []
    # Parcourir tous les fichiers dans le répertoire
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Vérifier que le fichier est un .txt
            file_path = os.path.join(directory_path, filename)
            print("file found:",filename)
            # Lire le fichier avec pandas
            #df = pd.read_csv(file_path, delimiter="\t", header=None)  # Ajuster le délimiteur si nécessaire
            df = pd.read_csv(file_path, delimiter=r"\s+", header=None)
            # Séparer les données (colonnes sauf la dernière) et les labels (dernière colonne)
            data = df.iloc[:, :-1].values   # Convertir les données en array
            data = StandardScaler().fit_transform(data)
            labels = df.iloc[:, -1].values  # Convertir les labels en array
            data, labels = resample_to_n(data, labels, n,min_class_size=10, 
                                         random_state=42, augmentation_std=0.1)
            # Ajouter les données et les labels aux listes
            data_list.append(data)
            labels_list.append(labels)
            files.append(os.path.splitext(filename)[0])
            sizes.append(len(data))
            p.append(data.shape[1])
            
    return files, sizes, p, data_list, labels_list

######################################################################################
def Load_reeldata(directory_path = "Highdimreelles",t = 10, start = 0):
    
    files, sizes, p,L_data, L_labels = load_classification_data_from_directory(directory_path)
    print(files, sizes,p)
    return files, L_data, L_labels
        
######################################################################################
'''
directory_path = "realdata"
Load_reeldata(directory_path = directory_path,t = 10, start = 0)
'''
#load_classification_data_from_directory(directory_path,1000)        