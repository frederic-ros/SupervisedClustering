# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:21:25 2025

@author: frederic.ros
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def plot_density_with_projection(X, densities, method="tsne", 
                                 title="Projection avec densité", cmap="viridis", random_state=42):
    """
    Projette les données X en 2D avec t-SNE ou UMAP, et affiche les points colorés selon leur densité.

    Parameters:
    - X : ndarray (N, D), données d'entrée
    - densities : ndarray (N,), densité associée à chaque point (valeurs entre 0 et 1)
    - method : str, "tsne" ou "umap"
    - title : str, titre du graphique
    - cmap : str, colormap matplotlib
    - random_state : int, pour la reproductibilité
    """
    if method == "tsne":
        projector = TSNE(n_components=2, random_state=random_state, perplexity=30)
    if method == "umap":
        projector = umap.UMAP(n_components=2, random_state=random_state)
    
    X_2d = projector.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=densities, cmap=cmap, s=30, edgecolor='k')
    plt.colorbar(scatter, label='Densité')
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel('Projection 1')
    plt.ylabel('Projection 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()