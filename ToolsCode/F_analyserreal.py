# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 07:15:20 2025

@author: frederic.ros
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def afficher_heatmap_resultats(matrice, nom_colonnes=None, nom_lignes=None, titre="Résultats des méthodes"):
    """
    matrice : array-like ou DataFrame de taille (n_datasets, n_méthodes)
              - Peut inclure une 1ère colonne pour les noms de datasets
    nom_colonnes : liste optionnelle des noms des méthodes
    nom_lignes : liste optionnelle des noms des datasets
    """

    # Si matrice est un DataFrame déjà bien formé, pas besoin de modifier
    if isinstance(matrice, pd.DataFrame):
        df = matrice.copy()
    else:
        df = pd.DataFrame(matrice)

    # Si la 1ère colonne contient les noms de datasets
    if nom_colonnes:
        df.columns = nom_colonnes
    if nom_lignes:
        df.index = nom_lignes

    # Création de la figure
    plt.figure(figsize=(14, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, cbar_kws={'label': 'Score'})
    plt.title(titre, fontsize=16)
    plt.ylabel("Datasets")
    plt.xlabel("Methods")
    plt.tight_layout()
    plt.show()

def construire_matrices_3_colonnes(repertoire):
    fichiers_txt = sorted([f for f in os.listdir(repertoire) if f.endswith('.txt')])
    donnees_col1, donnees_col2, donnees_col3 = [], [], []
    max_lignes = 0

    for fichier in fichiers_txt:
        chemin = os.path.join(repertoire, fichier)
        col1, col2, col3 = [], [], []

        with open(chemin, 'r') as f:
            for ligne in f:
                valeurs = ligne.strip().split('\t')
                try:
                    nums = [float(v) for v in valeurs]
                    if len(nums) >= 3:
                        col1.append(nums[0])
                        col2.append(nums[1])
                        col3.append(nums[2])
                except ValueError:
                    continue  # ligne non numérique

        donnees_col1.append(col1)
        donnees_col2.append(col2)
        donnees_col3.append(col3)
        max_lignes = max(max_lignes, len(col1))

    def to_matrix(donnees, max_len):
        mat = np.full((len(donnees), max_len), np.nan)
        for i, ligne in enumerate(donnees):
            mat[i, :len(ligne)] = ligne
        return mat

    matrice_1 = to_matrix(donnees_col1, max_lignes)
    matrice_2 = to_matrix(donnees_col2, max_lignes)
    matrice_3 = to_matrix(donnees_col3, max_lignes)

    return matrice_1, matrice_2, matrice_3

def pourcentage_lignes_nulles(repertoire):
    resultats = []

    fichiers_txt = [f for f in os.listdir(repertoire) if f.endswith('.txt')]
    
    for fichier in fichiers_txt:
        chemin_fichier = os.path.join(repertoire, fichier)
        total = 0
        lignes_nulles = 0

        with open(chemin_fichier, 'r') as f:
            for ligne in f:
                valeurs = ligne.strip().split('\t')
                try:
                    numeriques = [float(v) for v in valeurs]
                except ValueError:
                    continue  # ignorer les lignes non numériques

                total += 1
                if all(v == 0.0 for v in numeriques):
                    lignes_nulles += 1

        pourcentage = (lignes_nulles / total * 100) if total > 0 else 0
        resultats.append([fichier, round(100 - pourcentage, 2)])

    # Création du DataFrame
    df_nulles = pd.DataFrame(resultats, columns=["Fichier", "% Lignes nulles"])
    print(df_nulles)
    return df_nulles

# Exemple d’appel :
# df_nulles = pourcentage_lignes_nulles('/chemin/vers/ton/repertoire')

def df_to_latex_moy_std(df):
    # Identifier les colonnes de mesures
    colonnes = [col for col in df.columns if col != 'Fichier']
    
    # Regrouper les colonnes par paires (moyenne / écart type)
    colonnes_groupes = [colonnes[i:i+2] for i in range(0, len(colonnes), 2)]
    
    # Générer la nouvelle table avec format "moyenne (écart-type)"
    table_formatee = pd.DataFrame()
    table_formatee['Fichier'] = df['Fichier']
    
    for i, (col_m, col_s) in enumerate(colonnes_groupes, start=1):
        nom_col = f'Col{i}'
        table_formatee[nom_col] = df.apply(
            lambda row: f"{row[col_m]:.4f} ({row[col_s]:.4f})", axis=1
        )
    
    # Générer le tableau LaTeX
    latex_table = table_formatee.to_latex(index=False, escape=False, column_format='l' + 'c'*len(colonnes_groupes))
    
    return latex_table
def analyser_fichiers_par_fichier(repertoire):
    fichiers_txt = [f for f in os.listdir(repertoire) if f.endswith('.txt')]

    if not fichiers_txt:
        print("Aucun fichier .txt trouvé dans le répertoire.")
        return

    for fichier in fichiers_txt:
        chemin_fichier = os.path.join(repertoire, fichier)
        lignes_valides = []

        with open(chemin_fichier, 'r') as f:
            for ligne in f:
                valeurs = ligne.strip().split('\t')  # Séparateur tabulation
                try:
                    numeriques = [float(v) for v in valeurs]
                    if all(v == 0.0 for v in numeriques):
                        continue  # Ignore ligne de zéros
                    lignes_valides.append(numeriques)
                except ValueError:
                    continue  # Ignore les lignes non numériques

        if not lignes_valides:
            print(f"{fichier} : Aucune donnée valide.")
            continue

        data = np.array(lignes_valides)
        moyennes = np.mean(data, axis=0)
        ecarts_type = np.std(data, axis=0, ddof=1)  # ddof=1 = écart-type empirique

        print(f"\nFichier : {fichier}")
        print("Moyennes par colonne :", np.round(moyennes, 4))
        print("Écarts type par colonne :", np.round(ecarts_type, 4))

def analyser_fichiers_avec_tableau(repertoire):
    fichiers_txt = [f for f in os.listdir(repertoire) if f.endswith('.txt')]
    resume = []

    for fichier in fichiers_txt:
        chemin_fichier = os.path.join(repertoire, fichier)
        lignes_valides = []

        with open(chemin_fichier, 'r') as f:
            for ligne in f:
                valeurs = ligne.strip().split('\t')
                try:
                    numeriques = [float(v) for v in valeurs]
                    if all(v == 0.0 for v in numeriques):
                        continue
                    lignes_valides.append(numeriques)
                except ValueError:
                    continue

        if not lignes_valides:
            continue

        data = np.array(lignes_valides)
        moyennes = np.mean(data, axis=0)
        ecarts_type = np.std(data, axis=0, ddof=1)

        # Préparer la ligne du résumé : nom du fichier + moyennes + écarts type
        ligne = [fichier]
        for m, e in zip(moyennes, ecarts_type):
            ligne.extend([round(m, 4), round(e, 4)])
        resume.append(ligne)

    # Générer les noms de colonnes dynamiquement : col1_mean, col1_std, ...
    if resume:
        n_colonnes = (len(resume[0]) - 1) // 2
        noms_colonnes = ['Fichier']
        for i in range(1, n_colonnes + 1):
            noms_colonnes.append(f'Col{i}_moyenne')
            noms_colonnes.append(f'Col{i}_ecart_type')

        df_resume = pd.DataFrame(resume, columns=noms_colonnes)

        print("\nRésumé global :")
        print(df_resume)

        # Optionnel : sauvegarde en CSV
        # df_resume.to_csv("resume_statistiques.csv", index=False)

        return df_resume
    else:
        print("Aucune donnée valide dans les fichiers.")
        return None

# Exemple d'appel
r_f = "results/"
analyser_fichiers_par_fichier(r_f)
df = analyser_fichiers_avec_tableau(r_f)
#print(df)
dfnul = pourcentage_lignes_nulles(r_f)
print("dfnul",dfnul)

M1, M2, M3 = construire_matrices_3_colonnes(r_f)
M1 = M1.T
M2 = M2.T
M3 = M3.T
print (M3)
print("TAILLE matrices", M1.shape, M2.shape, M3.shape)

afficher_heatmap_resultats(M1, nom_colonnes=None, nom_lignes=None, titre="AMI")
afficher_heatmap_resultats(M2, nom_colonnes=None, nom_lignes=None, titre="ARI")
afficher_heatmap_resultats(M3, nom_colonnes=None, nom_lignes=None, titre="CVI")

'''
# Sauvegarde dans un fichier texte formaté
with open("recap", 'w') as f_out:
    f_out.write(df.to_string(index=False))

# Supposons que df_resume est déjà chargé avec tes données (ou lu via pd.read_csv)
latex_code = df_to_latex_moy_std(df)

# Affichage ou sauvegarde dans un fichier
print(latex_code)
'''