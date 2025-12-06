# -*- coding: utf-8 -*-
"""
Created on Fri May  2 07:01:45 2025

@author: frederic.ros
"""
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import umap
from F_affichagegenpoint import plot_pure_and_fuzzy_points_s

import numpy as np

def sample_k_patterns(input_patterns, output_patterns, k, random_state=None):
    """
    Sélectionne k éléments aléatoirement dans le dataset.

    Parameters
    ----------
    input_patterns : ndarray
        Matrice des patterns d'entrée (N, features)
    output_patterns : ndarray
        Matrice des targets (N, nb_classes)
    k : int
        Nombre d'éléments à sélectionner
    random_state : int, optional
        Graine pour reproductibilité

    Returns
    -------
    sampled_input : ndarray
        Patterns d'entrée sélectionnés
    sampled_output : ndarray
        Targets sélectionnés
    """
    rng = np.random.default_rng(random_state)
    N = input_patterns.shape[0]
    k = min(k, N)  # au cas où k > N
    selected_idx = rng.choice(N, size=k, replace=False)
    return input_patterns[selected_idx], output_patterns[selected_idx]

def compute_onlyinput_patterns_with_embedding(X, p_embedding=10, Kv_in=5,
                                              embedding_dict=None, random_state=None, normalize_distances=True
                                              ):
    """
    Génère les patterns d'entrée pour prédire les memberships à partir 
    d'une matrice KxK de distances symétrique + un embedding.

    Args:
        X (ndarray): Matrice de données (N, p_original)
        memberships (ndarray): Matrice d'appartenance (N, nb_classes)
        p_embedding (int): Taille de l'embedding
        Kv_in (int): Nombre de voisins (K)
        embedding_dict (dict): Dictionnaire des embeddings fixes selon p_original
        random_state (int): Graine aléatoire pour reproductibilité
        normalize_distances (bool): Si True, normalise les distances entre 0 et 1 pour chaque pattern

    Returns:
        input_patterns (ndarray): Matrice des patterns d'entrée (N, K^2 + p)
        output_patterns (ndarray): Matrice des targets (N, nb_classes)
    """
    if random_state is not None:
        np.random.seed(random_state)

    N, dim = X.shape
    input_patterns = []

    # Générer un embedding unique pour chaque p original si non fourni
    if embedding_dict is None:
        embedding_dict = {}

    if p_embedding not in embedding_dict:
        embedding_dict[p_embedding] = get_fixed_embedding(p_embedding)

    embedding = embedding_dict[p_embedding]

    # Recherche des Kv_in voisins les plus proches
    neigh = NearestNeighbors(n_neighbors=Kv_in, algorithm='auto').fit(X)
    _, indices_neighbors = neigh.kneighbors(X)

    for i in range(N):
        neighbors_idx = indices_neighbors[i]

        distances_matrix = np.zeros((Kv_in, Kv_in))

        # Distances entre l'item central et ses voisins (ligne 0)
        for j in range(Kv_in):
            distances_matrix[0, j] = np.linalg.norm(
                X[neighbors_idx[0]] - X[neighbors_idx[j]])

        # Distances entre les voisins entre eux
        for j in range(1, Kv_in):
            for k in range(Kv_in):
                distances_matrix[j, k] = np.linalg.norm(
                    X[neighbors_idx[j]] - X[neighbors_idx[k]])

        distances_flat = distances_matrix.flatten()
        distances_flat = distances_flat / np.sqrt(X.shape[1])

        #if normalize_distances:
        if normalize_distances:
            d_min, d_max = distances_flat.min(), distances_flat.max()
            if d_max > d_min:
                distances_flat = (distances_flat - d_min) / \
                    (d_max - d_min + 1e-8)
            else:
                # toutes les distances sont égales
                distances_flat = np.zeros_like(distances_flat)

        input_pattern = np.concatenate([distances_flat, embedding])
        input_patterns.append(input_pattern)

    return np.array(input_patterns)




def generate_distant_noise(Z, n_noise, dim, factor=3.0, scale=1.0, max_trials=10_000):
    """
    Génère des points de bruit éloignés d'un nuage de points existant.

    Parameters
    ----------
    Z : np.ndarray
        Points d'origine (N, dim)
    n_noise : int
        Nombre de points de bruit à générer
    dim : int
        Dimension de l'espace
    factor : float
        Facteur multiplicatif sur la distance moyenne au 1er voisin
    scale : float
        Amplitude de la zone d'échantillonnage
    max_trials : int
        Nombre maximum de tentatives de génération

    Returns
    -------
    np.ndarray
        Tableau des points de bruit générés (<= n_noise, dim)
    """
    tree = KDTree(Z)

    # Estimation de la distance moyenne au 1er voisin
    dists, _ = tree.query(Z, k=2)  # k=2 car le 1er voisin est le point lui-même
    avg_intra_dist = np.mean(dists[:, 1])

    distant_points = []
    trials = 0

    while len(distant_points) < n_noise and trials < max_trials:
        candidate = np.random.uniform(low=-scale, high=scale, size=(1, dim))
        dist, _ = tree.query(candidate, k=1)
        if dist[0][0] > factor * avg_intra_dist:
            distant_points.append(candidate[0])
        trials += 1

    return np.array(distant_points)

def filter_patterns(input_patterns, output_patterns, i_pure, i_noise, i_fuzzy):
    """
    Filtre les patterns selon les indices donnés pour 'i_pure', 'i_noise', et 'i_fuzzy'.
    Retourne un sous-ensemble des données avec n_pure == n_noise + n_fuzzy.

    Args:
    - input_patterns (ndarray): Matrice des patterns d'entrée (N, K^2 + p).
    - output_patterns (ndarray): Matrice des targets (N, nb_classes).
    - i_pure (ndarray): Indices des éléments 'purs'.
    - i_noise (ndarray): Indices des éléments 'bruyés'.
    - i_fuzzy (ndarray): Indices des éléments 'flous'.

    Returns:
    - filtered_input (ndarray): Sous-ensemble des patterns d'entrée filtrés.
    - filtered_output (ndarray): Sous-ensemble des targets filtrés.
    """
    # Nombre d'éléments à récupérer
    n_noise_fuzzy = len(i_noise)  # Le nombre d'éléments à prendre de i_pure

    # Randomiser les indices i_pure pour en sélectionner le même nombre que i_noise + i_fuzzy
    np.random.seed(42)  # Graine pour reproductibilité (modifie si besoin)
    #selected_pure_idx = np.random.choice(i_pure, n_noise_fuzzy, replace=False)
    
    n_noise_fuzzy = min(n_noise_fuzzy, len(i_pure))
    selected_pure_idx = np.random.choice(i_pure, (int)(n_noise_fuzzy/2), replace=False)
    # Combiner tous les indices
    selected_indices = np.concatenate([i_noise, i_fuzzy, selected_pure_idx])
    
    # Filtrer les patterns d'entrée et les targets
    filtered_input = input_patterns[selected_indices]
    filtered_output = output_patterns[selected_indices]

    return filtered_input, filtered_output


def get_fixed_embedding(p, size=10):
    """
    Retourne un embedding déterministe de taille `size` pour une valeur donnée de p.
    Utilise une fonction sinusoïdale sur un espace régulier.
    """
    rng = np.linspace(0, 1, size)
    # pattern déterministe dépendant de p
    embedding = np.sin(2 * np.pi * p * rng)
    return embedding


def compute_input_patterns_with_embedding(
    X, memberships, p_embedding=10, Kv_in=5,
    embedding_dict=None, random_state=None,
    normalize_distances=False
):
    """
    Génère les patterns d'entrée pour prédire les memberships à partir 
    d'une matrice KxK de distances symétrique + un embedding.

    Args:
        X (ndarray): Matrice de données (N, p_original)
        memberships (ndarray): Matrice d'appartenance (N, nb_classes)
        p_embedding (int): Taille de l'embedding
        Kv_in (int): Nombre de voisins (K)
        embedding_dict (dict): Dictionnaire des embeddings fixes selon p_original
        random_state (int): Graine aléatoire pour reproductibilité
        normalize_distances (bool): Si True, normalise les distances entre 0 et 1 pour chaque pattern

    Returns:
        input_patterns (ndarray): Matrice des patterns d'entrée (N, K^2 + p)
        output_patterns (ndarray): Matrice des targets (N, nb_classes)
    """
    if random_state is not None:
        np.random.seed(random_state)

    N, dim = X.shape
    input_patterns = []
    output_patterns = []

    # Générer un embedding unique pour chaque p original si non fourni
    if embedding_dict is None:
        embedding_dict = {}

    if p_embedding not in embedding_dict:
        embedding_dict[p_embedding] = get_fixed_embedding(p_embedding)

    embedding = embedding_dict[p_embedding]

    # Recherche des Kv_in voisins les plus proches
    neigh = NearestNeighbors(n_neighbors=Kv_in, algorithm='auto').fit(X)
    _, indices_neighbors = neigh.kneighbors(X)

    for i in range(N):
        neighbors_idx = indices_neighbors[i]

        distances_matrix = np.zeros((Kv_in, Kv_in))

        # Distances entre l'item central et ses voisins (ligne 0)
        for j in range(Kv_in):
            distances_matrix[0, j] = np.linalg.norm(
                X[neighbors_idx[0]] - X[neighbors_idx[j]])

        # Distances entre les voisins entre eux
        for j in range(1, Kv_in):
            for k in range(Kv_in):
                distances_matrix[j, k] = np.linalg.norm(
                    X[neighbors_idx[j]] - X[neighbors_idx[k]])

        distances_flat = distances_matrix.flatten()
        distances_flat = distances_flat / np.sqrt(X.shape[1])

        '''
        if normalize_distances:
            d_min, d_max = distances_flat.min(), distances_flat.max()
            if d_max > d_min:
                distances_flat = (distances_flat - d_min) / \
                    (d_max - d_min + 1e-8)
            
            #distances_flat = distances_flat / np.sqrt(dim)
        else:
                # toutes les distances sont égales
            distances_flat = np.zeros_like(distances_flat)
        '''
        
        input_pattern = np.concatenate([distances_flat, embedding])
        input_patterns.append(input_pattern)
        output_patterns.append(memberships[i])

    return np.array(input_patterns), np.array(output_patterns)



def extract_fuzzy_noise_pure_adaptive(membership_matrix, q_pure=0.85, q_noise=0.15, min_fraction=0.05):
    """
    Version adaptative de l'extraction des indices 'pure', 'noise' et 'fuzzy'.
    Les seuils sont calculés dynamiquement à partir de la distribution
    des valeurs de confiance (max des memberships par ligne).

    Args:
    - membership_matrix (ndarray): Matrice d'appartenance (N x nb_classes)
    - q_pure (float): Quantile supérieur pour définir les 'purs' (par défaut 0.85)
    - q_noise (float): Quantile inférieur pour définir les 'bruités' (par défaut 0.15)
    - min_fraction (float): Fraction minimale pour chaque catégorie (ex: 0.05 = 5%)

    Returns:
    - indices_pure (ndarray): Indices des points purs
    - indices_noise (ndarray): Indices des points bruités
    - indices_fuzzy (ndarray): Indices des points flous
    - thresholds (tuple): (threshold_noise, threshold_pure)
    """

    max_membership = np.max(membership_matrix, axis=1)
    N = len(max_membership)

    # Calcul des seuils adaptatifs
    threshold_pure = np.quantile(max_membership, q_pure)
    threshold_noise = np.quantile(max_membership, q_noise)

    # Sélection initiale
    indices_pure = np.where(max_membership >= threshold_pure)[0]
    indices_noise = np.where(max_membership <= threshold_noise)[0]
    indices_fuzzy = np.where((max_membership > threshold_noise) & (max_membership < threshold_pure))[0]

    # Vérification : éviter qu’un groupe soit trop petit
    min_count = int(min_fraction * N)

    if len(indices_pure) < min_count:
        # élargir la définition des purs
        threshold_pure = np.sort(max_membership)[-min_count]
        indices_pure = np.where(max_membership >= threshold_pure)[0]

    if len(indices_noise) < min_count:
        # élargir la définition des bruités
        threshold_noise = np.sort(max_membership)[min_count]
        indices_noise = np.where(max_membership <= threshold_noise)[0]

    indices_fuzzy = np.where((max_membership > threshold_noise) & (max_membership < threshold_pure))[0]

    return indices_pure, indices_noise, indices_fuzzy




def extract_fuzzy_noise_pure_k(membership_matrix, k_pure=None, k_noise=None):
    """
    Sélectionne les indices de points purs, bruités et flous en choisissant
    un nombre fixe d'éléments pour les catégories pure et noise.

    Args:
    - membership_matrix (ndarray): N x nb_classes
    - k_pure (int or None): nombre de points purs à extraire (les plus sûrs)
    - k_noise (int or None): nombre de points bruités à extraire (les moins sûrs)

    Returns:
    - indices_pure, indices_noise, indices_fuzzy (ndarray)
    """
    max_membership = np.max(membership_matrix, axis=1)
    N = len(max_membership)

    # trier indices par valeur de membership
    sorted_idx = np.argsort(max_membership)  # ordre croissant
#    indices_noise = sorted_idx[:k_noise] if k_noise is not None else np.array([], dtype=int)
    # Identifier les points bruités (membership < 0.05 pour toutes les classes)
    indices_noise = np.where(np.max(membership_matrix, axis=1) <= 0.05)
 
    indices_pure = sorted_idx[-k_pure:] if k_pure is not None else np.array([], dtype=int)

    # tous les autres deviennent flous
    mask = np.ones(N, dtype=bool)
    mask[indices_pure] = False
    mask[indices_noise] = False
    indices_fuzzy = np.where(mask)[0]

    return indices_pure, indices_noise, indices_fuzzy



def extract_fuzzy_noise_pure_topk(membership_matrix, k_pure=None, threshold_noise=0.05):
    """
    Sélectionne les indices des points purs, bruités et flous.
    - Les points bruités sont définis par un seuil.
    - Les points purs sont les k meilleurs (max membership).
    - Les autres points deviennent flous.

    Args:
    - membership_matrix (ndarray): N x nb_classes
    - k_pure (int or None): nombre de points purs à extraire parmi les meilleurs
    - threshold_noise (float): seuil pour considérer un point comme bruité

    Returns:
    - indices_pure, indices_noise, indices_fuzzy (ndarray)
    """
    max_membership = np.max(membership_matrix, axis=1)
    N = len(max_membership)

    # indices des points bruités
    indices_noise = np.where(max_membership <= threshold_noise)[0]

    # indices des purs : prendre les k meilleurs parmi le reste
    remaining_idx = np.setdiff1d(np.arange(N), indices_noise)
    if k_pure is not None:
        sorted_idx = remaining_idx[np.argsort(-max_membership[remaining_idx])]  # décroissant
        indices_pure = sorted_idx[:k_pure]
    else:
        # si k_pure None, prendre tous les points restants
        indices_pure = remaining_idx

    # les autres deviennent flous
    mask = np.ones(N, dtype=bool)
    mask[indices_pure] = False
    mask[indices_noise] = False
    indices_fuzzy = np.where(mask)[0]

    return indices_pure, indices_noise, indices_fuzzy


def extract_fuzzy_noise_pure(membership_matrix, threshold_pure=0.95, threshold_noise=0.05):
    """
    Calcule les indices des points purs, bruités et flous dans une matrice d'appartenance.

    Args:
    - membership_matrix (ndarray): Matrice d'appartenance (N x nb_classes), où chaque ligne est un vecteur de probabilité
    - threshold_pure (float): Seuil pour considérer un point comme "pur" (par défaut 0.95)
    - threshold_noise (float): Seuil pour considérer un point comme "bruit" (par défaut 0.05)

    Returns:
    - indices_pure (ndarray): Indices des points purs
    - indices_noise (ndarray): Indices des points bruités
    - indices_fuzzy (ndarray): Indices des points flous
    """

    # Identifier les points purs (membership > 0.95 pour l'une des classes)
    indices_pure = np.where(
        np.max(membership_matrix, axis=1) >= threshold_pure)[0]

    # Identifier les points bruités (membership < 0.05 pour toutes les classes)
    indices_noise = np.where(
        np.max(membership_matrix, axis=1) <= threshold_noise)[0]

    # Identifier les points flous (autres points)
    indices_fuzzy = np.where((np.max(membership_matrix, axis=1) < threshold_pure) &
                             (np.max(membership_matrix, axis=1) > threshold_noise))[0]

    return indices_pure, indices_noise, indices_fuzzy




def generate_distant_noise_improved(Z, n_noise, dim,
                                    factor=3.0, scale=1.0,
                                    batch_size=1024, max_trials=200,
                                    relax_factor=True, relax_step=0.9,
                                    fallback_topk_pool=10000, random_state=None):
    """
    Génère des points "distant noise" par rapport à Z de manière robuste en haute dimension.

    Principes :
    - construit un domaine d'échantillonnage basé sur l'étendue de Z (bounding box élargie)
      ou autour du centroïde si l'étendue est trop petite.
    - échantillonne en batch et sélectionne les candidats dont la distance au plus proche
      voisin de Z dépasse factor * avg_intra_dist.
    - si on n'obtient pas assez de candidats, on relâche progressivement le facteur (optionnel),
      ou on retourne les meilleurs candidats parmi un plus grand pool (fallback).

    Args:
        Z: array (n_samples, dim) - données de référence
        n_noise: int - nombre de points à générer
        dim: int - dimension (doit être = Z.shape[1])
        factor: float - multiplicateur de la distance moyenne intra-cluster
        scale: float - multiplicateur supplémentaire pour le bounding box
        batch_size: int - nombre de candidats générés par itération
        max_trials: int - nombre d'itérations batches avant fallback
        relax_factor: bool - si True, réduit `factor *= relax_step` quand trop strict
        relax_step: float - facteur multiplicatif (<1) pour relaxer
        fallback_topk_pool: int - taille du pool candidat pour fallback final
        random_state: int or None

    Returns:
        numpy array shape (n_noise, dim)
    """
    rng = np.random.default_rng(random_state)

    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] != dim:
        raise ValueError("Z must be shape (n_samples, dim) with given dim")

    tree = KDTree(Z)

    # distance moyenne au 1er vrai voisin
    dists, _ = tree.query(Z, k=2)  # k=2 car 1er nearest = point lui-même
    avg_intra_dist = float(np.mean(dists[:, 1]))

    # bounding box basé sur Z (étendue par feature)
    mins = Z.min(axis=0)
    maxs = Z.max(axis=0)
    ranges = maxs - mins

    # si range nul (ou très petit) sur certaines dims, utiliser centrage autour du centroïde
    if np.all(ranges == 0):
        center = Z.mean(axis=0)
        bbox_low = center - scale
        bbox_high = center + scale
    else:
        # étendre la bounding box par 'scale' multiplicatif (scale >= 1)
        bbox_low = mins - 0.5 * scale * ranges
        bbox_high = maxs + 0.5 * scale * ranges

        # si certaines composantes ont très petite étendue, garantir une largeur minimale
        min_width = 1e-6
        small = ranges < min_width
        if np.any(small):
            bbox_low[small] = mins[small] - scale
            bbox_high[small] = maxs[small] + scale

    distant_points = []
    candidates_pool = []  # pour fallback si besoin
    current_factor = factor

    trials = 0
    while len(distant_points) < n_noise and trials < max_trials:
        # échantillonnage uniforme dans la bounding box (batch)
        batch = rng.uniform(low=bbox_low, high=bbox_high, size=(batch_size, dim))
        d_batch, _ = tree.query(batch, k=1)
        d_batch = d_batch.ravel()
        mask = d_batch > (current_factor * avg_intra_dist)
        if mask.any():
            selected = batch[mask]
            for p in selected:
                distant_points.append(p)
                if len(distant_points) >= n_noise:
                    break

        # stocker candidates pour fallback
        candidates_pool.append((batch, d_batch))

        trials += 1
        # relaxer le critère si activé et si trop strict
        if relax_factor and trials % max(1, int(max_trials/5)) == 0:
            current_factor *= relax_step

    # Si on a obtenus suffisamment de points -> retour
    if len(distant_points) >= n_noise:
        return np.array(distant_points[:n_noise])

    # Fallback : rassembler un grand pool de candidats et retourner les top n_noise les plus éloignés
    # (utile dans les cas très stricts en haute dimension)
    pool_size = sum(b.shape[0] for b, _ in candidates_pool)
    if pool_size == 0:
        # En dernier recours, échantillonner autour du centroïde avec une grande variance
        center = Z.mean(axis=0)
        large_pool = rng.normal(loc=center, scale=5.0 * avg_intra_dist, size=(fallback_topk_pool, dim))
        d_large, _ = tree.query(large_pool, k=1)
        idx_top = np.argsort(-d_large.ravel())[:n_noise]
        return large_pool[idx_top]

    # concaténation des batches et distances
    all_batches = np.vstack([b for b, _ in candidates_pool])
    all_dists = np.hstack([d for _, d in candidates_pool])

    # prendre les top n_noise selon la distance
    idx_sorted = np.argsort(-all_dists)  # décroissant
    top_idx = idx_sorted[:min(n_noise, len(idx_sorted))]
    selected_points = all_batches[top_idx]

    # si encore pas assez (très improbable), compléter par échantillonnage gaussien très large
    if selected_points.shape[0] < n_noise:
        center = Z.mean(axis=0)
        extra = rng.normal(loc=center, scale=5.0 * avg_intra_dist, size=(n_noise - selected_points.shape[0], dim))
        selected_points = np.vstack([selected_points, extra])

    return selected_points[:n_noise]


def compute_membership_similarity_matrix(X, membership_matrix, K=5):
    """
    Calcule une matrice de similarité entre chaque point et ses K plus proches voisins
    à partir de la matrice des appartenances (membership_matrix).

    Args:
        X (np.ndarray): Données de forme (n_samples, n_features)
        membership_matrix (np.ndarray): Matrice (n_samples, n_clusters)
        K (int): Nombre de voisins à considérer (le premier est lui-même)

    Returns:
        similarity_matrix (np.ndarray): Matrice (n_samples, K) avec les scores de similarité
        neighbors_indices (np.ndarray): Indices des K plus proches voisins pour chaque point
    """
    neigh = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(X)
    _, indices = neigh.kneighbors(X)  # indices.shape = (n_samples, K)

    n_samples = X.shape[0]
    similarity_matrix = np.zeros((n_samples, K))

    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i]):
            sim = np.dot(membership_matrix[i], membership_matrix[j])
            similarity_matrix[i, j_idx] = sim

    return similarity_matrix, indices


def stack_X_and_memberships(X_pure, proba_pure,
                            X_fuzzy, proba_fuzzy,
                            X_noise, proba_noise):
    
    """
    Concatène les points et leurs matrices d'appartenance correspondantes.

    Args:
        X_pure (ndarray): Points purs (NxD)
        proba_pure (ndarray): Appartenances (N x n_classes)
        X_fuzzy (ndarray): Points flous (MxD)
        proba_fuzzy (ndarray): Appartenances (M x n_classes)
        X_noise (ndarray): Points de bruit (KxD)
        proba_noise (ndarray): Appartenances (K x n_classes) — généralement nuls

    Returns:
        X_all (ndarray): Tous les points concaténés (N+M+K, D)
        membership_matrix (ndarray): Matrice d'appartenance (N+M+K, n_classes)
    """
    if len(X_noise) > 0 and len(X_fuzzy) > 0:
        X_all = np.vstack([X_pure, X_fuzzy, X_noise])
        membership_matrix = np.vstack([proba_pure, proba_fuzzy, proba_noise])
    else:
        if len(X_fuzzy) > 0:
            X_all = np.vstack([X_pure, X_fuzzy])
            membership_matrix = np.vstack([proba_pure, proba_fuzzy])
        else:
            X_all = X_pure
            membership_matrix = proba_pure
    
    return X_all, membership_matrix


def compute_membership_matrix(X_pure, y_pure, X_fuzzy, clf):
    """
    Calcule la matrice d'appartenance pour tous les points (purs et flous),
    en utilisant le classifieur entraîné sur les points purs.

    Args:
        X_pure (ndarray): Points purs (NxD)
        y_pure (ndarray): Labels des points purs (N,)
        X_fuzzy (ndarray): Points flous ou bruités (MxD)
        clf (sklearn classifier): Classifieur entraîné

    Returns:
        membership_matrix (ndarray): Matrice d'appartenance (N+M, nb_classes)
        X_all (ndarray): Tous les points concaténés (N+M, D)
    """
    proba_fuzzy = []
    # Probabilités des points purs (connaissant leur label)
    proba_pure = clf.predict_proba(X_pure)

    # Probabilités des points flous (non supervisés)
    if len(X_fuzzy) > 0:
        proba_fuzzy = clf.predict_proba(X_fuzzy)

    return proba_pure, proba_fuzzy


def plot_pure_and_fuzzy_points(X_pure, y_pure, X_fuzzy, fuzzy_scores, Z_noise=[], title="Pure vs Fuzzy/Noisy Points"):
    """
    Affiche :
    - les points purs en couleur selon leur label,
    - les points flous/bruités en niveaux de gris selon leur appartenance moyenne.

    Args:
        X_pure (ndarray): Points purs (Nx2)
        y_pure (ndarray): Labels des points purs
        X_fuzzy (ndarray): Points flous ou bruités (Mx2)
        fuzzy_scores (ndarray): Score moyen d'appartenance pour chaque point flou (entre 0 et 1)
        title (str): Titre du graphique
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Affichage des points purs
    '''  
    ax.scatter(X_pure[:, 0], X_pure[:, 1], c=y_pure,
               cmap='tab10', edgecolors='k', s=50, label='Pure Points')
    '''
    ax.scatter(X_pure[:, 0], X_pure[:, 1], c=y_pure,
               cmap='tab10', edgecolors='k', s=50)
    # Affichage des points flous/bruités en niveaux de gris selon leur appartenance
    # 1 = blanc (flou), 0 = noir (fortement attaché)
    grey_levels = 1 - fuzzy_scores
    # grey_levels = 1.0 - np.clip(fuzzy_scores, 0, 1)  # s'assure que les valeurs sont dans [0, 1]
    '''
    ax.scatter(X_fuzzy[:, 0], X_fuzzy[:, 1], c="grey",
               marker='x', s=30, label='Fuzzy/Noisy Points')
    '''
    ax.scatter(X_fuzzy[:, 0], X_fuzzy[:, 1], c="grey",
               marker='x', s=30)
    
    if len(Z_noise) > 0:
        '''
        plt.scatter(Z_noise[:, 0], Z_noise[:, 1], c='k',
                    marker='x', s=30, label="Bruit")
        '''
        plt.scatter(Z_noise[:, 0], Z_noise[:, 1], c='k',
                    marker='x', s=30)

    ax.set_title(title)
    ax.legend()
    ax.axis('equal')
    plt.show()



def get_soft_memberships(X_pure, X, clf, K=5):
    """
    Calcule les appartenances soft aux clusters pour chaque point de X,
    en utilisant la moyenne des prédictions sur ses K plus proches voisins
    dans X_pure + X (incluant les points eux-mêmes).

    Parameters:
    - X_pure : ndarray, ensemble labellisé
    - X : ndarray, les points à évaluer
    - clf : classificateur entraîné sur X_pure
    - K : int, nombre de voisins à utiliser (excluant le point lui-même)

    Returns:
    - avg_proba : ndarray (N, n_classes), moyenne des probabilités pour chaque point de X
    """
    if len(X) == 0:
        return None

    # Construction du corpus complet pour la recherche des voisins
    corpus = np.vstack([X_pure, X])
    neigh = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(corpus)

    # Recherche des voisins pour chaque point de X
    _, indices = neigh.kneighbors(X)

    # On enlève le premier voisin (le point lui-même)
    neighbor_indices = indices[:, 1:]

    # Extraction des points voisins depuis le corpus complet
    # shape: (n_points, K, n_features)
    neighbor_points = corpus[neighbor_indices]

    # Prédictions des probabilités
    n_points = X.shape[0]
    # pour obtenir le nombre de classes
    n_classes = clf.predict_proba(X_pure[:1]).shape[1]
    all_proba = np.zeros((n_points, K, n_classes))

    for i in range(n_points):
        all_proba[i] = clf.predict_proba(neighbor_points[i])

    # Moyenne sur les voisins
    avg_proba = all_proba.mean(axis=1)

    return avg_proba




def plot_clusters_noisedistant(Z_clean, y_clean, Z_noise, title="Clusters avec bruit vrai"):
    """
    Affiche les clusters (avec labels différents de -1) et les points de bruit (label -1) en 2D.

    - Z : np.ndarray (n_samples, 2), les données.
    - y_pred : np.ndarray (n_samples,), labels (bruit = -1).
    - title : str, titre du graphique.
    """
    if Z_clean.shape[1] != 2:
        print("Cette fonction ne supporte que les données 2D.")
        #raise ValueError("Cette fonction ne supporte que les données 2D.")

    plt.figure(figsize=(6, 6))
    '''
    plt.scatter(Z_clean[:, 0], Z_clean[:, 1], c=y_clean,
                cmap='tab10', s=20, label="Clusters")
    '''
    plt.scatter(Z_clean[:, 0], Z_clean[:, 1], c=y_clean,
                cmap='tab10', s=20)
    if len(Z_noise) > 0:
        '''
        plt.scatter(Z_noise[:, 0], Z_noise[:, 1], c='k',
                    marker='x', s=30, label="Bruit")
        '''
        plt.scatter(Z_noise[:, 0], Z_noise[:, 1], c='k',
                    marker='x', s=30)
    plt.title(title, fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_clusters_with_noise(Z, y_pred, title="Clusters avec bruit détecté"):
    """
    Affiche les clusters (avec labels différents de -1) et les points de bruit (label -1) en 2D.

    - Z : np.ndarray (n_samples, 2), les données.
    - y_pred : np.ndarray (n_samples,), labels (bruit = -1).
    - title : str, titre du graphique.
    """
    if Z.shape[1] != 2:
#        raise ValueError("Cette fonction ne supporte que les données 2D.")
         print("Cette fonction ne supporte que les données 2D.")

    Z_clean = Z[y_pred != -1]
    y_clean = y_pred[y_pred != -1]
    Z_noise = Z[y_pred == -1]

    plt.figure(figsize=(6, 6))
    '''
    plt.scatter(Z_clean[:, 0], Z_clean[:, 1], c=y_clean,
                cmap='tab10', s=20, label="Clusters")
    '''
    plt.scatter(Z_clean[:, 0], Z_clean[:, 1], c=y_clean,
                cmap='tab10', s=20)
    if len(Z_noise) > 0:
        '''
        plt.scatter(Z_noise[:, 0], Z_noise[:, 1], c='k',
                    marker='x', s=30, label="Bruit")
        '''
        plt.scatter(Z_noise[:, 0], Z_noise[:, 1], c='k',
                    marker='x', s=30)
    plt.title(title, fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def get_soft_labels(X, y, clf, K=5, seuil=0.9):
    neigh = NearestNeighbors(n_neighbors=K+1).fit(X)
    _, indices = neigh.kneighbors(X)

    scores = []
    for i, idxs in enumerate(indices):
        neighbors = X[idxs[1:]]  # on ignore le point lui-même
        proba = clf.predict_proba(neighbors)
        proba_same = [p[int(y[i])] for p in proba]
        avg_proba = np.mean(proba_same)
        scores.append(avg_proba)

    scores = np.array(scores)

    # Filtrage des points avec confiance > seuil
    mask = scores > seuil
    Z = X[mask]
    y_filtered = y[mask]

    return scores, Z, y_filtered



def get_soft_labels_fast(X, y, clf, K=5, seuil=0.9):
    """
    Calcule la probabilité moyenne de la classe vraie de chaque point sur ses K voisins.
    Filtre les points dont la moyenne est supérieure à `seuil`.
    
    Parameters:
        X : np.ndarray, shape (N, dim)
            Points à traiter
        y : np.ndarray, shape (N,)
            Labels des points
        clf : classifieur sklearn déjà entraîné
            Doit implémenter predict_proba
        K : int
            Nombre de voisins à considérer
        seuil : float
            Seuil de filtrage sur la probabilité moyenne
    Returns:
        avg_proba : np.ndarray, shape (N,)
            Probabilité moyenne de la classe vraie
        Z : np.ndarray
            Points filtrés
        y_filtered : np.ndarray
            Labels filtrés
    """
    # K plus 1 pour exclure le point lui-même
    neigh = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(X)
    _, indices = neigh.kneighbors(X)
    neighbor_indices = indices[:, 1:]  # exclut le point lui-même
    flat_indices = neighbor_indices.ravel()
    neighbor_points = X[flat_indices]

    # Probabilités prédites pour tous les voisins
    all_proba = clf.predict_proba(neighbor_points)
    n_points = X.shape[0]
    n_classes = all_proba.shape[1]
    all_proba = all_proba.reshape((n_points, K, n_classes))

    # Mapping des labels y vers les indices dans clf.classes_
    # On filtre aussi les points dont la classe n'est pas vue par clf
    mask_valid = np.isin(y, clf.classes_)
    if not np.all(mask_valid):
        X_valid = X[mask_valid]
        y_valid = y[mask_valid]
        all_proba = all_proba[mask_valid]
        y = y_valid
    else:
        X_valid = X

    class_indices = np.array([np.where(clf.classes_ == yi)[0][0] for yi in y])
    y_indices = class_indices[:, np.newaxis, np.newaxis]  # (N_valid, 1, 1)

    # Probabilité de la classe vraie pour chaque voisin
    proba_true_class = np.take_along_axis(all_proba, y_indices, axis=2).squeeze(2)  # (N_valid, K)
    avg_proba = proba_true_class.mean(axis=1)  # moyenne sur K voisins

    # Filtrage
    mask = avg_proba > seuil
    Z = X_valid[mask]
    y_filtered = y[mask]

    return avg_proba, Z, y_filtered



def get_pure_and_fuzzy_points(X, y, clf, K=5, seuil=0.9, use_fuzzy=True):
    """
    Identifie les points 'purs' et 'fuzzy' selon la cohérence
    des K plus proches voisins et les probabilités du classifieur.

    Paramètres
    ----------
    X : array-like (n_samples, n_features)
        Données d'entrée.
    y : array-like (n_samples,)
        Labels réels.
    clf : modèle sklearn entraîné avec predict_proba()
        Classifieur pour estimer la cohérence des voisins.
    K : int
        Nombre de voisins à considérer (défaut=5).
    seuil : float
        Seuil de pureté moyen (défaut=0.9).
    use_fuzzy : bool
        Si False, X et y ne sont pas modifiés et les ensembles fuzzy sont vides.
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    if not use_fuzzy:
        # Pas de traitement -> on renvoie X et y inchangés
        fuzzy_points = np.empty((0, X.shape[1]))
        fuzzy_labels = np.empty((0,), dtype=y.dtype)
        return X, y, fuzzy_points, fuzzy_labels

    # --- sinon, on applique tout le traitement ---
    neigh = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(X)
    _, indices = neigh.kneighbors(X)

    neighbor_indices = indices[:, 1:]
    flat_indices = neighbor_indices.ravel()
    neighbor_points = X[flat_indices]

    all_proba = clf.predict_proba(neighbor_points)
    n_points = X.shape[0]
    n_classes = all_proba.shape[1]
    all_proba = all_proba.reshape((n_points, K, n_classes))

    # On ne garde que les points dont la classe existe dans clf
    mask_valid = np.isin(y, clf.classes_)
    X_valid = X[mask_valid]
    y_valid = y[mask_valid]
    all_proba = all_proba[mask_valid]

    # Probabilité moyenne d'appartenance à la classe réelle
    class_indices = np.array([np.where(clf.classes_ == yi)[0][0] for yi in y_valid])
    y_indices = class_indices[:, np.newaxis, np.newaxis]
    proba_true_class = np.take_along_axis(all_proba, y_indices, axis=2).squeeze(2)
    avg_proba = proba_true_class.mean(axis=1)

    # Séparation pure/fuzzy
    mask_pure = avg_proba > seuil
    mask_fuzzy = ~mask_pure

    pure_points = X_valid[mask_pure]
    pure_labels = y_valid[mask_pure]
    fuzzy_points = X_valid[mask_fuzzy]
    fuzzy_labels = y_valid[mask_fuzzy]

    return pure_points, pure_labels, fuzzy_points, fuzzy_labels

def plot_soft_labels(X, scores, titre="Soft labels (membership degree)"):
    scores = np.array(scores)
    norm_scores = (scores - scores.min()) / \
        (scores.max() - scores.min() + 1e-8)

    if X.shape[1] == 2:
        plt.figure()
        plt.title(titre, fontsize=10)
        plt.scatter(X[:, 0], X[:, 1], c=norm_scores, s=10, cmap='viridis')
        #plt.colorbar(label="Score d'appartenance")
        plt.axis('equal')
        #plt.grid(True)
        plt.show()

    elif X.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                        c=norm_scores, cmap='viridis', s=10)
        #fig.colorbar(sc, label="Score d'appartenance")
        fig.colorbar(sc)
        ax.set_title(titre, fontsize=10)
        plt.show()

    else:
        print("Visualisation uniquement pour les données 2D ou 3D.")


def drawcluster2D3D(X, y, titre=""):

    y = np.array(y)
    if X.shape[1] == 2:
        plt.figure()
        plt.title(titre, fontsize=10)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap='tab10')
        plt.axis('equal')
    #    plt.grid(True)
        plt.show()
    elif X.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(titre, fontsize=10)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=10, cmap='tab10')
        plt.show()


def getT(dim):
    # Matrice de rotation/anisotropie aléatoire
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    return Q


def random_rotation(dim):
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    return Q


def getT_anisotropic(dim, anisotropy_factor=3.0):
    U, _, _ = np.linalg.svd(np.random.randn(dim, dim))
    D = np.diag(np.linspace(1.0, anisotropy_factor, dim))
    return U @ D


def Hypercube(dim, scale=1.0):
    # Retourne les sommets d'un hypercube centré en 0
    return np.array([[scale * (2 * ((i >> j) & 1) - 1) for j in range(dim)] for i in range(2**dim)])

def sample_unique_hypercube_vertices(dim, n_centers, scale=1.0):
    """
    Tire n_centers sommets distincts de l'hypercube {−scale, +scale}^dim
    sans générer les 2**dim sommets entiers.
    """
    if n_centers > 2**dim:
        raise ValueError("Impossible : n_centers > 2**dim")

    rng = np.random.default_rng()
    seen = set()
    centers = []

    while len(centers) < n_centers:
        # Génère un sommet binaire aléatoire
        vertex = rng.integers(0, 2, size=dim)
        tup = tuple(vertex)
        if tup not in seen:
            seen.add(tup)
            # Convertit {0,1} → {−scale, +scale}
            centers.append(scale * (2 * vertex - 1))

    return np.array(centers)



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


def CreateData(n_samples=256, dim=2, hamming_distance=0.1,p_noise=0.1, noise_t=1.0,
               max_dev=0.5, n_centers=4, Score_min=0.9,
               Draw=0, Transform_each_cluster=True,
               add_perturbation=True, perturbation_amplitude=0.02,
               min_cluster_size=30):

    noise_only = noise_distant = []
    scale = 1.0
    n_noise_full = int(n_samples * p_noise)
    n_points = n_samples - n_noise_full
 
    n_noise_in = (int)(n_noise_full / 2) #by default 0.5
    
    
    # Centres sur les sommets d'un hypercube   
    max_cluster = min(n_centers, 2**dim)
    
    Centers = sample_unique_hypercube_vertices_h(dim, max_cluster, scale=scale, 
                                                 min_hamming_ratio=hamming_distance)
    #Centers = sample_unique_hypercube_vertices(dim, max_cluster, scale=scale)   
 
    # Répartition des tailles de cluster avec contrainte
    S = np.random.rand(n_centers)
    S = S / S.sum()
    S = (S * (n_points - n_centers * min_cluster_size)
         ).astype(int) + min_cluster_size

    while S.sum() < n_points:
        S[np.random.randint(0, n_centers)] += 1
    while S.sum() > n_points:
        i = np.random.randint(0, n_centers)
        if S[i] > min_cluster_size:
            S[i] -= 1
            
    rng = np.random.default_rng()  # Nouveau générateur (non seedé)
    # Écarts-types pour chaque cluster
    std = rng.uniform(scale * 0.05, scale *
                            max_dev, size=(n_centers, dim))
   
    #print("STD",std)

    all_points = []
    all_labels = []

    for i, center in enumerate(Centers):
        # Points gaussiens
        pts = np.random.randn(S[i], dim) * std[i]

        # Transformation anisotropique
        if Transform_each_cluster:
            factor = np.random.uniform(1.5, 4.0)
            T = getT_anisotropic(dim, anisotropy_factor=factor)
            pts = pts @ T  # anisotropie appliquée ici
            R = random_rotation(dim)
            pts = pts @ R
            # Encadrement dans la boîte [-0.5, 0.5]^dim (avant translation)
            max_abs = np.max(np.abs(pts), axis=0)
            scale_factor = 1.1 / (max_abs + 1e-8)  # évite la division par 0
            pts = pts * scale_factor  # contenu dans [-0.5, 0.5] environ
        # Translation vers le centre du cluster
        pts += center

        # Perturbation locale
        if add_perturbation:
            eps = perturbation_amplitude * scale
            pts += np.random.uniform(-eps, eps, size=pts.shape)

        all_points.append(pts)
        all_labels.append(np.full(S[i], i))

    Z = np.vstack(all_points)
    y = np.concatenate(all_labels)

    if Draw:
        drawcluster2D3D(Z, y, titre="Initial clusters")

    # Classification KNN
    k = max(3, int(np.sqrt(len(Z)) // 2))

    knn = KNeighborsClassifier(n_neighbors=k)

    y_p = y.copy()
    score = 0
    while (score < 0.999):
        knn.fit(Z, y_p)
        y_pred = knn.predict(Z)
        score = accuracy_score(y_p, y_pred)
        y_p = y_pred
  
    if Draw:
        drawcluster2D3D(Z, y_pred, titre="With classifier corrections")


    if n_noise_full > 0:
        noise_threshold = 0.8
        n_noise = n_noise_in
        # Génération de points de bruit (dans l'hypercube étendu)
        noise = np.random.uniform(
            low=-1.0*scale, high=1.0*scale, size=(n_noise, dim)) * noise_t

        
        # Classification avec probabilités
        probas = knn.predict_proba(noise) #je classifie les bruits injectés.
        confidence = probas.max(axis=1) #je cherche la proba max.
        labels = probas.argmax(axis=1) #je cherche le label correspondant.

        # Seuil d'acceptation, je prends les labels ou - 1 suivant le seuil.
        final_labels = np.where(confidence >= noise_threshold, labels, -1)
         
        # Ajout des points bruités à la base
        Z1 = np.vstack([Z, noise])
        y1_pred = np.concatenate([y_pred, final_labels])
        noise_only = noise[final_labels == -1] #ceux sont les points noise reels.
        
        
        # Séparation des points classés et non classés (bruit)
        mask_classified = final_labels != -1  #je prends les non bruits.
        classified_noise = noise[mask_classified]
        classified_labels = final_labels[mask_classified]

         
        # Mise à jour de Z et y_pred avec uniquement les points bruités bien classifiés? On retourne Z!!
        Z = np.vstack([Z, classified_noise]) #vérifies.
        y_pred = np.concatenate([y_pred, classified_labels])
        

        if Draw:
            plot_clusters_with_noise(
                Z1, y1_pred, title="Clusters with internal noise")
        #il faut que la taille de Z + noiseonly + noisedistant =1000     
        #n_noise_true = n_noise_full - n_noise #ancienne version
        n_noise_true = n_samples - len(Z) - len(noise_only) #nouvelle version
        
        #print("true,noise only,not noise",n_noise_true,len(noise_only),len(not_noise))
        #noise_distant = generate_distant_noise(Z,n_noise=n_noise_true,dim=dim,factor=3.0)
        
        noise_distant = generate_distant_noise_improved(Z,              # tes données de référence
                                                  n_noise=n_noise_true,    # nombre de points de bruit à générer
                                                  dim=dim,         # dimension de l’espace (doit correspondre à Z.shape[1])
                                                  factor=3.0,     # écart minimum au voisin le plus proche (3× la distance moyenne)
                                                  scale=1.5,      # élargit la zone d’échantillonnage autour de Z
                                                  random_state=42 # pour reproductibilité
                                                  )
        
        if Draw:
            plot_clusters_noisedistant(Z, y_pred, noise_distant, title="Clusters with distant noisy points")

    #print("Z,noise only,noise true,noise distant",len(Z),len(noise_only),n_noise_true,len(noise_distant))
    return Z, y_pred, noise_only, noise_distant

import itertools

def sample_hamming_clusters(dim, n_centers, max_hamming=2):
    """
    Génère n_centers sommets distincts de {0,1}^dim.
    - On choisit un centre aléatoire.
    - On prend tous les sommets à distance de Hamming <= max_hamming autour de lui.
    - Si on n'a pas assez de centres, on choisit un nouveau centre éloigné des précédents.
    """
    rng = np.random.default_rng()
    seen = set()
    centers = []

    while len(centers) < n_centers:
        # Trouver un centre non encore visité
        for _ in range(1000):
            center = rng.integers(0, 2, size=dim)
            tup = tuple(center)
            if tup not in seen:
                break

        # Générer les sommets dans la boule de Hamming autour de ce centre
        local_points = [center]
        for d in range(1, max_hamming + 1):
            for comb in itertools.combinations(range(dim), d):
                new = center.copy()
                new[list(comb)] = 1 - new[list(comb)]
                tup_new = tuple(new)
                if tup_new not in seen:
                    local_points.append(new)

        rng.shuffle(local_points)

        # Ajouter autant de points que nécessaire sans dépasser n_centers
        for p in local_points:
            tup_p = tuple(p)
            if tup_p not in seen:
                centers.append(p)
                seen.add(tup_p)
                if len(centers) >= n_centers:
                    break

    return np.array(centers)

def k_distance_stats(X, k=8):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)  # +1 car le point lui-même
    dists, idx = nbrs.kneighbors(X)
    # ignore self-distance (col 0)
    k_dists = dists[:, -1]   # distance au k-ième voisin
    return {
        "mean_k_dist": np.mean(k_dists),
        "std_k_dist": np.std(k_dists),
        "min_k_dist": np.min(k_dists),
        "max_k_dist": np.max(k_dists),
        "cv_k_dist": np.std(k_dists)/np.mean(k_dists)
     }

def genpoint(n_samples=1000, dim=2, max_dev=0.1, p_noise=0.1,hamming_distance=0.01, 
             p_embedding=10, Kv_in=5,
             Kv_out=16, K_m=5, n_centers = 4, filtered=True, 
             Draw=False, save=False, use_fuzzy=True):

    # len(X) + len(noise_points) + len(noise_distant) = n_samples
    X, y, noise_points, noise_distant = CreateData(n_samples=n_samples, dim=dim,  
                                                   n_centers=n_centers,
                                                   p_noise=p_noise, max_dev=max_dev,
                                                   hamming_distance=hamming_distance,
                                                   add_perturbation=True,
                                                   perturbation_amplitude=0.05,
                                                   Transform_each_cluster=True, Draw=Draw,
                                                   min_cluster_size=100)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if Draw == 1:
        drawcluster2D3D(X, y, titre="Clusters without noisy points")
    if len(noise_points) > 0:
        noise_points = scaler.transform(noise_points)
    if len(noise_distant) > 0:
        noise_distant = scaler.transform(noise_distant)
    C = len(np.unique(y))
    #clf = RandomForestClassifier(n_estimators=100)
    k = max(3, int(np.sqrt(len(X)) // 2))
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    # Scores d'appartenance: scores Z, y_filtered les points en confiance.
    scores, Z, y_filtered = get_soft_labels_fast(X, y, clf=clf, K=K_m, seuil=0.9)
    
    # On va trier les pure points et les pures labels.
    pure_points, pure_labels, fuzzy_points, fuzzy_labels = get_pure_and_fuzzy_points(
        X, y, clf=clf, K=K_m, seuil=0.9,use_fuzzy=use_fuzzy)
    
    
    '''
    if Draw == 1:
        drawcluster2D3D(pure_points, pure_labels, titre="Pure clusters")
        plot_soft_labels(X, scores)
        drawcluster2D3D(Z, y_filtered, titre="Pure clusters(Z and y filterered)")
    '''
    
    if len(noise_points) > 0:
        Z_fuzzy = np.vstack([fuzzy_points, noise_points])
    else:
        Z_fuzzy = fuzzy_points
      
    #vérification affichage ???
    if Draw == 1:
        plot_pure_and_fuzzy_points_s(X_pure=Z, y_pure=y_filtered,
                                   X_fuzzy=Z_fuzzy,
                                   Z_noise=noise_distant,
                                   title="Clusters and fuzzy/noisy points",
                                   reduce_dim=True, method='umap')
    
    # on a Z et y filtered (c'est du sur sur les labels.) => on calcule la membership matrix
    #pour les Z et les Z fuzzy(fuzzy point + bruit interne)
    
    proba_pure, proba_fuzzy = compute_membership_matrix(Z, y_filtered, Z_fuzzy, clf)
    
    #maintenant on met tous dans membership matrixe
    if len(noise_distant) > 0:
        M = noise_distant.shape[0]
        memberships_noise = np.zeros((M, C))
        X_all, membership_matrix = stack_X_and_memberships(Z, proba_pure, Z_fuzzy,
                                                           proba_fuzzy, noise_distant,
                                                           memberships_noise)
    else:
        X_all, membership_matrix = stack_X_and_memberships(Z, proba_pure, Z_fuzzy,
                                                           proba_fuzzy, [],
                                                           [])

    #la on calcule la similarité à partir de la membership matrix.
    sim_matrix, neighbors = compute_membership_similarity_matrix(
        X_all, membership_matrix, K=Kv_out)
    
    
    
    # extraction des indices purs, noise et fuzzy. (IDENTIQUE)
    i_pure, i_noise, i_fuzzy = extract_fuzzy_noise_pure(sim_matrix, threshold_pure=0.95,
                                                        threshold_noise=0.05)

    in_pat, out_pat = compute_input_patterns_with_embedding(X_all, sim_matrix,
                                                            p_embedding=p_embedding,
                                                            normalize_distances=False, Kv_in=Kv_in)

    if save == True:
        np.savetxt("genfichiers/membership_matrix.txt", membership_matrix,
                   delimiter='\t', fmt="%.6f")
        np.savetxt("genfichiers/X_all.txt", X_all, delimiter='\t', fmt="%.6f")
        np.savetxt("genfichiers/similarity_matrix.txt", sim_matrix,
                   fmt="%.4f", delimiter="\t")
        np.savetxt("genfichiers/neighbors_indices.txt", neighbors,
                   fmt="%d", delimiter="\t")
        np.savetxt("genfichiers/input_patterns.txt", in_pat, delimiter="\t", fmt="%.6f")
        np.savetxt("genfichiers/output_patterns.txt", out_pat, delimiter="\t", fmt="%.6f")

    if filtered == True:
        #in_pat, out_pat = sample_k_patterns(in_pat, out_pat, k=50, random_state=42)
        in_pat, out_pat = filter_patterns(in_pat, out_pat, i_pure, i_noise, i_fuzzy)
    return X_all, in_pat, out_pat


'''
genpoint(n_samples=1000, dim=2, max_dev=0.1,p_noise=0.1, p_embedding = 10, Kv_in = 5,
             Kv_out = 16, K_m = 5, filtered = True, Draw = True, save=False)
'''
