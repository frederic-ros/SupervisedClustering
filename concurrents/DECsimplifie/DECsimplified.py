# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:43:57 2025

@author: frederic.ros
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import numpy as np


# =====================
# 1️⃣  Autoencoder
# =====================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# =====================
# 2️⃣  Phase pré-entraînement de l’AE
# =====================
def pretrain_autoencoder(X, input_dim, latent_dim=10, epochs=100, lr=1e-3):
    model = Autoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        X_hat, _ = model(X_t)
        loss = criterion(X_hat, X_t)
        loss.backward()
        optimizer.step()
        '''
        if (epoch + 1) % 20 == 0:
            print(f"[Pretrain] Epoch {epoch+1}/{epochs} | Loss = {loss.item():.4f}")
        '''
    return model


# =====================
# 3️⃣  Phase clustering (DEC simplifié)
# =====================

def deep_embedding_clustering(
        X, y=None, n_clusters_low=2, n_clusters_up=9,
        latent_dim=10, epochs=100):

  
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pré-entraînement de l'autoencodeur
    model = pretrain_autoencoder(
        X_scaled, input_dim=X.shape[1],
        latent_dim=latent_dim, epochs=epochs
    )

    # Encodage latent
    model.eval()
    with torch.no_grad():
        _, Z = model(torch.tensor(X_scaled, dtype=torch.float32))
    Z = Z.numpy()

    # Recherche du meilleur nombre de clusters
    best_n = None
    best_sil = -1

    # IMPORTANT : tester jusqu'à n_clusters_up INCLUS
    for n_clusters in range(n_clusters_low, n_clusters_up + 1):

        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        y_pred_tmp = kmeans.fit_predict(Z)

        sil = silhouette_score(Z, y_pred_tmp)

        # Détection clusters trop petits
        unique, counts = np.unique(y_pred_tmp, return_counts=True)
        threshold = max(20, 0.1 * len(y_pred_tmp))
        large_clusters = [k for k, v in zip(unique, counts) if v >= threshold]

        # Mise à jour du meilleur modèle
        if sil > best_sil and len(large_clusters) >= 2:
            best_sil = sil
            best_n = n_clusters

    # Application finale avec best_n
    if best_n is not None:
        kmeans = KMeans(n_clusters=best_n, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(Z)
    else:
        # Cas dégénéré
        y_pred = np.zeros(len(Z), dtype=int)
        best_n = 1

    
    return Z, y_pred

'''
# =====================
# 4️⃣ Exemple d’utilisation
# =====================
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target

    Z, y_pred = deep_embedding_clustering(X, y, n_clusters=3, latent_dim=5, epochs=100)
'''