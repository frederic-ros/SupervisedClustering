# -*- coding: utf-8 -*-
"""
Created on Tue May  6 06:00:59 2025

@author: frederic.ros
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap
    has_umap = True
except ImportError:
    has_umap = False

def visualize_clusters_2d(data, labels, method='tsne',name=None):
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

    for label in unique_labels:
        cluster_points = data_2d[np.array(labels) == label]
        if label == -1:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color='gray', label='Noise')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_colors[label], label=f'Cluster {label}')

    if name == None: 
        plt.title("2D Visualization of Clusters"+" (D-space="+str(data.shape[1])+")")
    else: 
        plt.title(name)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
#    plt.legend()
    plt.show()


def cluster_mutual_density_from_density(data, densities, Kv=8, min_density=0.1):
    """
    Clusterisation fondée sur les connexions mutuelles et une densité minimale.

    Parameters:
    - data : ndarray (N, D), données originales
    - densities : ndarray (N,), densité associée à chaque point (pré-calculée)
    - Kv : int, nombre de voisins à considérer
    - min_density : float, densité minimale pour qu’un point participe au cluster

    Returns:
    - labels : ndarray (N,), étiquettes de cluster
    - G : graphe des connexions mutuelles
    """
    import numpy as np
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors

    n = data.shape[0]
    nbrs = NearestNeighbors(n_neighbors=Kv, algorithm='auto').fit(data)
    knn_indices = nbrs.kneighbors(data, return_distance=False)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Construire le graphe avec les connexions mutuelles et densité suffisante
    for i in range(n):
        neighbors_i = knn_indices[i]
        for j in neighbors_i:
            if i == j:
                continue
            neighbors_j = knn_indices[j]
            if i in neighbors_j:
                if densities[i] >= min_density and densities[j] >= min_density:
                    G.add_edge(i, j)

    # Clustering type DBSCAN
    visited = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)
    current_cluster = 0

    for i in np.argsort(-densities):  # Trie décroissant
        if visited[i] or densities[i] < min_density:
            continue

        cluster_queue = [i]
        labels[i] = current_cluster
        visited[i] = True

        while cluster_queue:
            node = cluster_queue.pop()
            for neighbor in G.neighbors(node):
                if not visited[neighbor]:
                    if densities[neighbor] >= min_density:
                        cluster_queue.append(neighbor)
                        labels[neighbor] = current_cluster
                    else:
                        labels[neighbor]


'''
def cluster_mutual_density(data, M_pred, Kv=8, threshold=0.5, min_density=3, normalize=True):
    n = data.shape[0]

    # Étape 1 : calcul des indices des K voisins
    nbrs = NearestNeighbors(n_neighbors=Kv, algorithm='auto').fit(data)
    knn_indices = nbrs.kneighbors(data, return_distance=False)

    # Étape 2 : construire graphe avec arêtes mutuelles confirmées
    G = nx.Graph()
    G.add_nodes_from(range(n))
    densities = np.zeros(n)

    for i in range(n):
        neighbors_i = knn_indices[i]
        for j_pos, j in enumerate(neighbors_i):
            if i == j:
                continue
            neighbors_j = knn_indices[j]
            # Trouver l'index de i dans les voisins de j
            if i in neighbors_j:
                j_pos_back = np.where(neighbors_j == i)[0]
                if len(j_pos_back) == 0:
                    continue

                pij = M_pred[i][j_pos]
                pji = M_pred[j][j_pos_back[0]]
                if pij >= threshold and pji >= threshold:
                    G.add_edge(i, j)
                    densities[i] += pij  # Somme des prédictions pour la densité
                    densities[j] += pji  # Somme des prédictions pour la densité

    # Normalisation des densités par rapport à Kv (si demandé)
    if normalize:
        densities = densities / (2*Kv)  # Normalisation par rapport au nombre de voisins Kv

    # Étape 3 : trier les nœuds par densité décroissante
    sorted_nodes = np.argsort(-densities)

    # Étape 4 : clustering type DBSCAN
    visited = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)  # -1 = bruit
    current_cluster = 0

    for i in sorted_nodes:
        if visited[i]:
            continue
        if densities[i] < min_density:
            continue

        # Expansion à partir du noyau
        cluster_queue = [i]
        labels[i] = current_cluster
        visited[i] = True

        while cluster_queue:
            node = cluster_queue.pop()
            for neighbor in G.neighbors(node):
                if not visited[neighbor]:
                    if densities[neighbor] >= min_density:
                        cluster_queue.append(neighbor)
                        labels[neighbor] = current_cluster
                    elif labels[neighbor] == -1:
                        # voisin faiblement dense : border point
                        labels[neighbor] = current_cluster
                    visited[neighbor] = True
        current_cluster += 1

    return labels, densities, G
'''

def cluster_mutual_density(data, M_pred, Kv=8, threshold=0.5, min_density=3, 
                    normalize=True, mutual=True):
    """
    Clustering basé sur les densités avec option mutualité.

    Args:
        data: np.array, shape (n_samples, n_features)
        M_pred: np.array, shape (n_samples, Kv), score de probabilité [0,1] pour chaque voisin
        Kv: int, nombre de voisins
        threshold: float, seuil pour créer une arête
        min_density: float, densité minimale pour initier un cluster
        normalize: bool, normaliser les densités
        mutual: bool, si True, utiliser la mutualité pour créer les arêtes

    Returns:
        labels: np.array, labels de clusters (-1 = bruit)
        densities: np.array, densité calculée pour chaque nœud
        G: networkx.Graph, graphe construit
    """
    n = data.shape[0]

    # Étape 1 : calcul des indices des K voisins
    nbrs = NearestNeighbors(n_neighbors=Kv, algorithm='auto').fit(data)
    knn_indices = nbrs.kneighbors(data, return_distance=False)

    # Étape 2 : construire le graphe
    G = nx.Graph()
    G.add_nodes_from(range(n))
    densities = np.zeros(n)

    for i in range(n):
        neighbors_i = knn_indices[i]
        for j_pos, j in enumerate(neighbors_i):
            if i == j:
                continue
            pij = M_pred[i][j_pos]

            if mutual:
                # mutualité : vérifier que i est dans les voisins de j
                neighbors_j = knn_indices[j]
                if i in neighbors_j:
                    j_pos_back = np.where(neighbors_j == i)[0][0]
                    pji = M_pred[j][j_pos_back]
                    if pij >= threshold and pji >= threshold:
                        G.add_edge(i, j)
                        densities[i] += pij
                        densities[j] += pji
            else:
                # non-mutual : créer l'arête si pij dépasse le seuil
                if pij >= threshold:
                    G.add_edge(i, j)
                    densities[i] += pij
                    densities[j] += pij

    # Normalisation
    if normalize:
        densities = densities / (2*Kv)

    # Étapes 3 & 4 : clustering type DBSCAN
    sorted_nodes = np.argsort(-densities)
    visited = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)
    current_cluster = 0

    for i in sorted_nodes:
        if visited[i]:
            continue
        if densities[i] < min_density:
            continue
        cluster_queue = [i]
        labels[i] = current_cluster
        visited[i] = True

        while cluster_queue:
            node = cluster_queue.pop()
            for neighbor in G.neighbors(node):
                if not visited[neighbor]:
                    if densities[neighbor] >= min_density:
                        cluster_queue.append(neighbor)
                        labels[neighbor] = current_cluster
                    elif labels[neighbor] == -1:
                        labels[neighbor] = current_cluster
                    visited[neighbor] = True
        current_cluster += 1

    return labels, densities, G
