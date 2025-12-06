# -*- coding: utf-8 -*-
"""
Created on Thu May  8 06:53:40 2025

@author: frederic.ros
"""
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def generate_heterogeneous_clusters(n_samples=1000, p_noise=0.05, n_clusters=3, delta_density=2.0,
                                     min_cluster_size=30, proj_dim=2, random_state=42):
    """
    Génère un jeu de données avec des clusters séparables ayant des tailles et densités très différentes,
    plus un bruit uniforme.
    
    Paramètres :
        n_samples : nombre total de points (y compris le bruit)
        p_noise : proportion de bruit
        n_clusters : nombre de clusters
        delta_density : variation de densité entre les clusters
        min_cluster_size : nombre minimal de points dans chaque cluster
        proj_dim : dimension d’espace de sortie (>=2)
        random_state : graine aléatoire
    
    Retour :
        X : données projetées (n_samples, proj_dim)
        y : étiquettes (-1 pour bruit)
    """
    np.random.seed(random_state)

    n_noise = int(p_noise * n_samples)
    n_clustered = n_samples - n_noise

    # Répartition aléatoire mais contrainte : chaque cluster a au moins min_cluster_size
    remaining = n_clustered - min_cluster_size * n_clusters
    random_weights = np.random.dirichlet(np.ones(n_clusters))
    raw_sizes = (random_weights * remaining).astype(int) + min_cluster_size

    # Ajustement final pour obtenir exactement n_clustered
    raw_sizes[-1] += n_clustered - raw_sizes.sum()

    # Densités décroissantes (écarts-types croissants)
    std_devs = np.linspace(1.0 / delta_density, 1.0, n_clusters)

    centers = np.random.uniform(-10, 10, size=(n_clusters, 2))
    X_clusters, y_clusters = [], []

    for i, (size, std) in enumerate(zip(raw_sizes, std_devs)):
        Xi, _ = make_blobs(n_samples=size, centers=[centers[i]], cluster_std=std, random_state=random_state + i)
        X_clusters.append(Xi)
        y_clusters.append(np.full(size, i))

    X = np.vstack(X_clusters)
    y = np.concatenate(y_clusters)

    # Ajout de bruit
    if n_noise > 0:
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_noise = np.random.uniform(low=X_min, high=X_max, size=(n_noise, X.shape[1]))
        y_noise = np.full(n_noise, -1)
        X = np.vstack([X, X_noise])
        y = np.concatenate([y, y_noise])

    # Normalisation
    X = StandardScaler().fit_transform(X)

    # Projection
    if proj_dim > 2:
        W = np.random.randn(2, proj_dim)
        X = X @ W

    return X, y


def generate_clusters_with_different_densities(n_samples=1000, p_noise=0.05, n_clusters=3, delta_density=2.0, proj_dim=5, random_state=42):
    """
    Génère des clusters séparables avec différentes densités, et ajoute un bruit uniforme.
    Paramètres:
        n_samples: Nombre total de points (incluant le bruit)
        p_noise: Proportion de bruit dans les données
        n_clusters: Nombre de clusters
        delta_density: Facteur qui contrôle la différence de densité entre les clusters
        proj_dim: Dimensionnalité de l'espace projeté
        random_state: Pour la reproductibilité
    Retour:
        X, y: Données générées et leurs labels
    """
    np.random.seed(random_state)

    # Nombre de points bruyants
    n_noise = int(p_noise * n_samples)
    n_clustered = n_samples - n_noise
    
    # Répartition des points dans les clusters
    cluster_sizes = np.random.randint(10, n_clustered // n_clusters, size=n_clusters)
    cluster_sizes = np.append(cluster_sizes, n_clustered - cluster_sizes.sum())  # ajuster la taille totale

    # Création des centres et des densités des clusters
    centers = np.random.uniform(-10, 10, size=(n_clusters, 2))
    std_devs = np.linspace(1.0, 1.0 / delta_density, n_clusters)  # variation de densité

    # Création des clusters
    X_clusters = []
    y_clusters = []
    for i, (size, std) in enumerate(zip(cluster_sizes, std_devs)):
        X_i, _ = make_blobs(n_samples=size, centers=[centers[i]], cluster_std=std, random_state=random_state+i)
        X_clusters.append(X_i)
        y_clusters.append(np.full(size, i))

    # Fusionner les clusters
    X = np.vstack(X_clusters)
    y = np.concatenate(y_clusters)

    # Ajout de bruit (bruit uniforme)
    if n_noise > 0:
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_noise = np.random.uniform(low=X_min, high=X_max, size=(n_noise, X.shape[1]))
        y_noise = np.full(n_noise, -1)  # -1 pour le bruit
        X = np.vstack([X, X_noise])
        y = np.concatenate([y, y_noise])

    # Normalisation des données
    X = StandardScaler().fit_transform(X)

    # Projection dans une plus grande dimension
    if proj_dim > 2:
        W = np.random.randn(2, proj_dim)
        X = X @ W

    return X, y
def generate_mixed_clusters_separable(n_samples_per_type=300, proj_dim=2, 
                                      random_state=42):
    np.random.seed(random_state)

    # Blobs (réduction de l'écart-type)
    X1, y1 = make_blobs(n_samples=n_samples_per_type, centers=3, cluster_std=0.5, random_state=random_state)

    # Moons (moins de bruit)
    X2, y2 = make_moons(n_samples=n_samples_per_type, noise=0.05)
    y2 += y1.max() + 1


    # Assemblage et projection
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    X = StandardScaler().fit_transform(X)

    if proj_dim > 2:
        W = np.random.randn(2, proj_dim)
        X = X @ W

    return X, y, y
'''
def generate_mixed_clusters_separable_1(n_samples_per_type=300, proj_dim=2, random_state=42):
    np.random.seed(random_state)

    # Blobs classiques
    X1, y1 = make_blobs(n_samples=n_samples_per_type, centers=3, cluster_std=0.5, random_state=random_state)

    # Moons : agrandis, tournés et déplacés
    X2, y2 = make_moons(n_samples=n_samples_per_type, noise=0.05)
    theta = np.pi / 4  # rotation de 45 degrés
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    X2 = X2 @ rotation_matrix        # rotation
    X2 = X2 * 2.5                    # scaling (agrandir les lunes)
    X2 = X2 + np.array([6, 6])       # décalage spatial

    y2 += y1.max() + 1  # rendre les labels disjoints

    # Assemblage
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    X = StandardScaler().fit_transform(X)

    # Projection si besoin
    if proj_dim > 2:
        W = np.random.randn(2, proj_dim)
        X = X @ W

    return X, y, y
'''
'''
def generate_mixed_clusters_separable_1(n_items=1000, proj_dim=2, p_noise=0.05, random_state=42):
    np.random.seed(random_state)

    # Calcul du nombre de points de bruit et de clusters
    n_noise = int(p_noise * n_items)
    n_clustered = n_items - n_noise
    n_types = 2  # blobs + moons
    n_samples_per_type = n_clustered // n_types

    # Blobs
    X1, y1 = make_blobs(n_samples=n_samples_per_type, centers=3, cluster_std=0.5, random_state=random_state)

    # Moons ajustés
    X2, y2 = make_moons(n_samples=n_samples_per_type, noise=0.05)
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    X2 = X2 @ rot
    X2 *= 2.5
    X2 += np.array([6, 6])
    y2 += y1.max() + 1

    # Données initiales
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    # Ajout du bruit uniforme
    if n_noise > 0:
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_noise = np.random.uniform(low=X_min, high=X_max, size=(n_noise, X.shape[1]))
        y_noise = np.full(n_noise, -1)  # Label -1 pour le bruit
        X = np.vstack([X, X_noise])
        y = np.concatenate([y, y_noise])

    # Normalisation globale
    X = StandardScaler().fit_transform(X)

    # Projection si besoin
    if proj_dim > 2:
        W = np.random.randn(2, proj_dim)
        X = X @ W

    return X, y, y
'''
def generate_mixed_clusters_separable_1(n_items=1000, proj_dim=2, p_noise=0.05, 
                                      min_blob_size=10, random_state=42):
    np.random.seed(random_state)

    n_noise = int(p_noise * n_items)
    n_clustered = n_items - n_noise
    n_types = 2  # blobs + moons
    n_clusters_blob = 3

    # Nombre total pour les blobs
    n_blob_total = n_clustered // n_types

    # Répartition aléatoire avec minimum garanti
    remaining = n_blob_total - min_blob_size * n_clusters_blob
    if remaining < 0:
        raise ValueError(f"Pas assez de points pour garantir min_blob_size={min_blob_size} par cluster.")
    rand_weights = np.random.dirichlet(np.ones(n_clusters_blob))
    extra_points = (rand_weights * remaining).astype(int)
    extra_points[0] += (remaining - extra_points.sum())  # ajustement exact

    n_blob_per_cluster = min_blob_size + extra_points

    # Std aléatoire
    blob_std = np.random.uniform(0.3, 1.0, size=n_clusters_blob)

    X_blob_list, y_blob_list = [], []
    start_label = 0
    for i, (n, std) in enumerate(zip(n_blob_per_cluster, blob_std)):
        Xi, _ = make_blobs(n_samples=n, centers=1, cluster_std=std, random_state=random_state+i)
        X_blob_list.append(Xi)
        y_blob_list.append(np.full(n, start_label + i))
    X1 = np.vstack(X_blob_list)
    y1 = np.concatenate(y_blob_list)

    # Moons ajustés
    n_moons = n_clustered - X1.shape[0]
    X2, y2 = make_moons(n_samples=n_moons, noise=0.05)
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    X2 = X2 @ rot
    X2 *= 2.5
    X2 += np.array([6, 6])
    y2 += y1.max() + 1

    # Fusion
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    # Bruit
    if n_noise > 0:
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_noise = np.random.uniform(low=X_min, high=X_max, size=(n_noise, X.shape[1]))
        y_noise = np.full(n_noise, -1)
        X = np.vstack([X, X_noise])
        y = np.concatenate([y, y_noise])

    # Normalisation
    X = StandardScaler().fit_transform(X)

    # Projection éventuelle
    if proj_dim > 2:
        W = np.random.randn(2, proj_dim)
        X = X @ W

    return X, y, y

def plot_2d(X, labels, title="2D Cluster Visualization"):
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20, alpha=0.8)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.colorbar(scatter, label="Label")
    plt.tight_layout()
    plt.show()

'''
for i in range(5):
    X, y, labels = generate_mixed_clusters_separable_1(n_items=1000, proj_dim=2, p_noise=0.05, 
                                                        random_state=42+i)
    plot_2d(X, labels, title="Clusters générés (labels)")
    print("taille",len(X))
'''    
from sklearn.preprocessing import StandardScaler

def generate_clusters(n_samples=1000, n_clusters=4, dim=10, separability=1.0, random_state=None):
    """
    Génère des clusters gaussiens séparables dans un espace de dimension quelconque.

    Paramètres
    ----------
    n_samples : int
        Nombre total d'échantillons.
    n_clusters : int
        Nombre de clusters.
    dim : int
        Dimension de l’espace.
    separability : float
        Contrôle la distance entre les centres (plus grand = clusters plus séparés).
        Typiquement entre 0.1 et 5.0.
    random_state : int ou None
        Graine aléatoire pour reproductibilité.

    Retour
    ------
    Z : ndarray (n_samples, dim)
        Les données simulées.
    y : ndarray (n_samples,)
        Les étiquettes de clusters correspondantes.
    """

    rng = np.random.default_rng(random_state)

    # --- Détermination du nombre de points par cluster ---
    base = np.ones(n_clusters)
    base = base / base.sum()
    counts = np.floor(base * n_samples).astype(int)
    counts[-1] += n_samples - counts.sum()  # ajustement

    # --- Génération des centres ---
    centers = rng.uniform(-separability, separability, size=(n_clusters, dim))

    # --- Génération des clusters ---
    all_points = []
    all_labels = []

    for i in range(n_clusters):
        # Écart-type proportionnel à la séparabilité inverse
        cluster_std = 0.2 / separability if separability > 0 else 0.2

        # Chaque cluster est une gaussienne autour de son centre
        pts = centers[i] + rng.normal(0, cluster_std, size=(counts[i], dim))
        all_points.append(pts)
        all_labels.append(np.full(counts[i], i))

    Z = np.vstack(all_points)
    y = np.concatenate(all_labels)

    # --- Mise à l’échelle ---
    Z = StandardScaler().fit_transform(Z)

    return Z, y

def generate_nonlinear_clusters(n_samples=1000, n_clusters=4, dim=10,
                                separability=1.0, nonlinearity=1.0, random_state=42):
    """
    Génère des clusters non linéaires en dimension élevée.
    Chaque cluster est une déformation non linéaire d’un petit nuage gaussien.
    """
    np.random.seed(random_state)
    
    samples_per_cluster = n_samples // n_clusters
    X_list, y_list = [], []

    for i in range(n_clusters):
        # --- Centre du cluster dans un hypercube ---
        center = np.random.uniform(-separability, separability, size=(dim,))

        # --- Nuage initial (gaussien) ---
        X = np.random.randn(samples_per_cluster, dim) * 0.5 + center

        # --- Déformation non linéaire (sinusoïde + rotation locale) ---
        if nonlinearity > 0:
            noise = np.random.randn(samples_per_cluster, dim) * 0.05
            X_nl = X + nonlinearity * np.sin(X) + noise
        else:
            X_nl = X

        X_list.append(X_nl)
        y_list.append(np.full(samples_per_cluster, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # --- Normalisation ---
    X = StandardScaler().fit_transform(X)

    return X, y
