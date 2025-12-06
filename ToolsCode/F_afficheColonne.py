# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 07:58:03 2025

@author: frederic.ros
"""
import numpy as np
import matplotlib.pyplot as plt


import numpy as np


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_columns(filename, title =None, columns=[0], labels=None):
    """
    Affiche graphiquement les valeurs d'une ou plusieurs colonnes d'un fichier .txt tabulé.

    Paramètres
    ----------
    filename : str
        Chemin du fichier texte (valeurs séparées par des tabulations).
    columns : list of int
        Liste des indices de colonnes à afficher (ex: [0, 2, 5]).
    labels : list of str, optionnel
        Noms personnalisés pour les colonnes sélectionnées.
        Si None, les colonnes seront nommées automatiquement ("Colonne {i}").
    """
    # Lecture du fichier
    data = np.loadtxt(filename, delimiter='\t')

    # Vérification du nombre de colonnes
    n_cols = data.shape[1]
    for c in columns:
        if c < 0 or c >= n_cols:
            raise ValueError(f"❌ La colonne {c} n'existe pas (le fichier contient {n_cols} colonnes).")

    # Création des abscisses (commence à 1)
    x = np.arange(1, data.shape[0] + 1)

    # Gestion des labels
    if labels is None:
        labels = [f"Colonne {c}" for c in columns]
    elif len(labels) != len(columns):
        raise ValueError("❌ Le nombre de labels doit correspondre au nombre de colonnes sélectionnées.")

    # Tracé
    plt.figure(figsize=(10, 5))
    for c, lbl in zip(columns, labels):
        plt.plot(x, data[:, c], marker='o', label=lbl)

    plt.title(title)
#    plt.xlabel("Index (début à 1)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Afficher toutes les abscisses
    plt.xticks(x)
    plt.tight_layout()
    plt.show()


    
#plot_columns("resultsreal/highdim/dim32/dsprites32methode14.txt", title="dsprites (D=32)",columns=[0,1,2], labels=["NMI","ARI","CVI"])
plot_columns("../resultsreal/highdim/dim32/minst32methode14aff.txt", title="MINST (D=32)",columns=[0,1,2], labels=["NMI","ARI","CVI"])
#plot_columns("resultsreal/highdim/dim100/dsprites100methode14.txt", title="dsprites (D=100)", columns=[0,1,2], labels=["NMI","ARI","CVI"])
plot_columns("../resultsreal/highdim/dim100/minst100methode14aff.txt", title="MINST (D=100)",columns=[0,1,2], labels=["NMI","ARI","CVI"])