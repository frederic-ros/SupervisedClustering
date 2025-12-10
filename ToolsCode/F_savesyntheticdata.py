# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 07:22:03 2025

@author: frederic.ros
"""

import numpy as np

def save_xy_to_txt(X, y, filename="dataset.txt"):
    """
    Sauvegarde X et y dans un fichier .txt avec tabulation comme séparateur.
    Chaque ligne = colonnes de X + label y.

    Paramètres :
        X : array (n_samples, n_features)
        y : array (n_samples,)
        filename : nom du fichier de sortie
    """
    # Vérification des dimensions
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X et y doivent avoir le même nombre d'échantillons.")

    # Concaténation des données
    data = np.hstack((X, y.reshape(-1, 1)))

    # Sauvegarde avec séparateur tabulation
    np.savetxt(
        filename,
        data,
        delimiter="\t",
        fmt="%.6f"  # format numérique, modifiable si besoin
    )

    print(f"Fichier sauvegardé : {filename}")
