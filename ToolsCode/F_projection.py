# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:25:28 2025

@author: frederic.ros
"""

import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import umap

def process_folder(folder_path, method="umap", n_components=2):
    """
    Parameters
    ----------
    folder_path : str
        Path to folder containing .txt files.
    method : str
        "umap" or "tsne".
    n_components : int
        Dimension of the reduced space.
    """

    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    files.sort()

    for fname in files:
        print(f"Processing {fname}...")

        # --- Load file ---
        full_path = os.path.join(folder_path, fname)
        df = pd.read_csv(full_path, sep="\t", header=None)

        # Separate features and label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)

        # --- Normalize features column-wise ---
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1e-6  # avoid division by zero
        X_norm = (X - mean) / std

        # --- Projection ---
        if method.lower() == "umap":
            reducer = umap.UMAP(n_components=n_components)
        else:
            reducer = TSNE(n_components=n_components)

        X_proj = reducer.fit_transform(X_norm)

        # --- Rebuild output ---
        output = np.hstack([X_proj, y])

        # --- Build output filename ---
        out_name = "P" + fname
        out_path = os.path.join(folder_path, out_name)

        # --- Save ---
        np.savetxt(out_path, output, fmt="%.6f", delimiter="\t")

        print(f"Saved: {out_path}")


# Exemple d'appel :
process_folder("test", method="umap", n_components=32)
