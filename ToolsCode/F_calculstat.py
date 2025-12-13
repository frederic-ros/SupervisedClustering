# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 06:33:59 2025

@author: frederic.ros
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_columns_from_txt_folder(folder_path, columns_to_plot=None, x_start=2):
    """
    Lit tous les fichiers .txt d'un dossier, extrait les valeurs par colonne,
    puis trace les courbes pour les colonnes sélectionnées.

    Parameters:
        folder_path (str): dossier contenant les fichiers .txt
        columns_to_plot (list[int] or None):
            - Liste des index de colonnes à afficher (1-based)
            - Si None : affiche toutes les colonnes
        x_start (int): valeur de départ de l'axe des abscisses (défaut = 2)
    """

    all_rows = []

    # Récupérer les valeurs de tous les fichiers
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as f:
                values = f.read().strip().split()

            row = [float(v) for v in values]
            all_rows.append(row)

    if len(all_rows) == 0:
        raise ValueError("Aucun fichier .txt trouvé dans le dossier.")

    data = np.array(all_rows)  # shape: (n_files, n_cols)
    n_cols = data.shape[1]

    # Si aucune colonne précisée → toutes
    if columns_to_plot is None:
        columns_to_plot = list(range(1, n_cols + 1))

    # Passage en index 0-based
    columns_to_plot = [c - 1 for c in columns_to_plot]

    # Vérification validité colonnes
    for c in columns_to_plot:
        if c < 0 or c >= n_cols:
            raise ValueError(f"Colonne {c+1} invalide. Votre fichier contient {n_cols} colonnes.")

    # Construction de l'axe des X
    x = np.arange(len(data)) + x_start

    # Tracé
    plt.figure(figsize=(10, 5))

    for c in columns_to_plot:
        plt.plot(x, data[:, c], marker='o', label=f"Colonne {c+1}")

    plt.title("Évolution des valeurs par colonne")
    plt.xlabel("Index des fichiers")
    plt.ylabel("Valeurs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_column_stats_from_txt_folder(folder_path):
    """
    Parcourt un dossier, lit tous les fichiers .txt contenant des valeurs tabulées
    et retourne les moyennes et écarts-types colonne par colonne.

    Parameters:
        folder_path (str): chemin du répertoire

    Returns:
        col_means (np.ndarray): moyennes par colonne
        col_stds (np.ndarray): écarts-types par colonne
    """
    
    all_rows = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Lecture du fichier (supposé sur une seule ligne)
            with open(file_path, "r") as f:
                values = f.read().strip().split()

            row = [float(v) for v in values]
            all_rows.append(row)

    if len(all_rows) == 0:
        raise ValueError("Aucun fichier .txt ou aucune donnée trouvée.")

    # Convertit en matrice NxK
    arr = np.array(all_rows)

    # Moyenne et écart type par colonne
    col_means = arr.mean(axis=0)
    col_stds = arr.std(axis=0)

    return col_means, col_stds


folder = "../resultsynthetique/all/KMEANSresultfolder3"
#folder = "../resultsynthetique/all"
mean_val, std_val = compute_column_stats_from_txt_folder(folder)
print("Moyenne :", mean_val)
print("Écart-type :", std_val)
plot_columns_from_txt_folder(folder, columns_to_plot=[5,6],x_start=2)
