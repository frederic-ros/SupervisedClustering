# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 10:14:22 2025

@author: frederic.ros
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib.image as mpimg

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def reduce_to_2d(*arrays, method='tsne', random_state=42):
    combined = np.vstack(arrays)
    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    else:
        reducer = TSNE(n_components=2, random_state=random_state)
    
    reduced = reducer.fit_transform(combined)
    split_indices = np.cumsum([arr.shape[0] for arr in arrays[:-1]])
    return np.split(reduced, split_indices)

def plot_pure_and_fuzzy_points_s(X_pure, y_pure, X_fuzzy, 
                                 Z_noise=[], 
                                 title="Pure vs Fuzzy/Noisy Points", reduce_dim=True, method='tsne'):
    """
    Affiche :
    - les points purs en couleur selon leur label,
    - les points flous/bruités en niveaux de gris selon leur appartenance moyenne.

    Réduction automatique de dimension si > 2.

    Args:
        X_pure (ndarray): Points purs (NxD)
        y_pure (ndarray): Labels des points purs
        X_fuzzy (ndarray): Points flous ou bruités (MxD)
        fuzzy_scores (ndarray): Score moyen d'appartenance (entre 0 et 1)
        Z_noise (ndarray): Points bruyants (KxD)
        title (str): Titre du graphique
        reduce_dim (bool): Réduire la dimension si >2
        method (str): 'tsne' ou 'umap'
    """
    all_arrays = [X_pure, X_fuzzy] + ([Z_noise] if len(Z_noise) > 0 else [])
    
    if X_pure.shape[1] > 2 and reduce_dim:
        X_pure, X_fuzzy, *rest = reduce_to_2d(*all_arrays, method=method)
        Z_noise = rest[0] if rest else []

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    ax.scatter(X_pure[:, 0], X_pure[:, 1], c=y_pure,
               cmap='tab10', edgecolors='k', s=50)

    #grey_levels = 1 - fuzzy_scores
    ax.scatter(X_fuzzy[:, 0], X_fuzzy[:, 1], c="grey",
               marker='x', s=30)

    if len(Z_noise) > 0:
        plt.scatter(Z_noise[:, 0], Z_noise[:, 1], c='k',
                    marker='x', s=30)

    ax.set_title(title, fontsize=10)
    #ax.legend(["Pure Points", "Fuzzy/Noisy Points", "Noise"] if len(Z_noise) > 0 else ["Pure Points", "Fuzzy/Noisy Points"])
    ax.axis('equal')
    plt.show()


import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize

def concat_images_in_grid(n, p, image_dir, output_path="concat_result.jpg"):
    """
    Concatène les n*p premières images d’un dossier en une grille (n lignes, p colonnes),
    redimensionne automatiquement si les tailles diffèrent, et sauvegarde le tout en JPEG.

    Args:
        n (int): Nombre de lignes
        p (int): Nombre de colonnes
        image_dir (str): Répertoire contenant les images
        output_path (str): Chemin de sortie de l’image concaténée (.jpg)
    """
    files = sorted([f for f in os.listdir(image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    num_required = n * p
    assert len(files) >= num_required, f"Pas assez d'images : il en faut {num_required}"

    # Charger les images
    images = []
    base_shape = None

    for idx, fname in enumerate(files[:num_required]):
        path = os.path.join(image_dir, fname)
        img = mpimg.imread(path)

        # Convertir RGBA → RGB
        if img.shape[-1] == 4:
            img = img[:, :, :3]

        # S'assurer que les valeurs sont dans [0,1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if base_shape is None:
            base_shape = img.shape[:2]  # hauteur, largeur
        else:
            # Redimensionner si taille différente
            if img.shape[:2] != base_shape:
                img = resize(img, base_shape, preserve_range=True, anti_aliasing=True)

        images.append(img)

    # Construire la grille
    rows = []
    for i in range(n):
        row = np.concatenate(images[i * p:(i + 1) * p], axis=1)
        rows.append(row)
    grid_image = np.concatenate(rows, axis=0)

    # Sauvegarde en JPEG
    plt.imsave(output_path, grid_image)
    print(f"Image sauvegardée sous : {output_path}")


#concat_images_in_grid(3, 3,"ImageGenerationArticle",  output_path="resultat.png")
#pour le dataset sans discovery
'''
concat_images_in_grid(1, 3,"realgraphe2",  output_path="resultat2.png")
concat_images_in_grid(1, 2,"realgraphe1",  output_path="resultat1.png")
'''
'''
concat_images_in_grid(1, 4,"realgraphe3",  output_path="resultat3.png")
concat_images_in_grid(1, 4,"realgraphe4",  output_path="resultat4.png")
'''
#concat_images_in_grid(3, 4,"synthetique2",  output_path="synthetic2.png")