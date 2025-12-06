# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:46:45 2025

@author: frederic.ros
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import load_iris


# ======================
# 1️⃣ VAE avec GMM latent
# ======================
class VaDE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, n_clusters=10):
        super(VaDE, self).__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # ---- Encoder (x → z_mean, z_log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.z_mean = nn.Linear(32, latent_dim)
        self.z_log_var = nn.Linear(32, latent_dim)

        # ---- Decoder (z → x_hat)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        # ---- Paramètres du GMM latent
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)
        self.mu = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.log_var = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def encode(self, x):
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(0.5 * log_var)

    def decode(self, z):
        return self.decoder(z)

    def compute_gamma(self, z):
        """Responsabilité du cluster pour chaque point."""
        z_expand = z.unsqueeze(1)  # (N, 1, D)
        mu_expand = self.mu.unsqueeze(0)  # (1, K, D)
        log_var_expand = self.log_var.unsqueeze(0)  # (1, K, D)

        log_prob = -0.5 * torch.sum(
            torch.exp(-log_var_expand) * (z_expand - mu_expand) ** 2 + log_var_expand, dim=2
        )
        log_prob += torch.log(self.pi + 1e-8)
        gamma = torch.softmax(log_prob, dim=1)
        return gamma

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decode(z)
        gamma = self.compute_gamma(z)
        return x_hat, z, z_mean, z_log_var, gamma


# ======================
# 2️⃣ Fonction de perte VaDE
# ======================
def vade_loss(x, x_hat, z, z_mean, z_log_var, gamma, model):
    """Combinaison reconstruction + KL divergence pondérée par les clusters."""
    # Reconstruction loss
    recon_loss = torch.mean((x - x_hat) ** 2)

    # KL divergence entre q(z|x) et p(z)
    z_expand = z.unsqueeze(1)
    mu_expand = model.mu.unsqueeze(0)
    log_var_expand = model.log_var.unsqueeze(0)

    p_c_z = torch.sum(gamma * (
        0.5 * torch.sum(log_var_expand + torch.exp(-log_var_expand) *
                        (z_expand - mu_expand) ** 2, dim=2)
    ), dim=1)

    kl_loss = torch.mean(p_c_z - torch.sum(0.5 * (1 + z_log_var), dim=1))

    return recon_loss + kl_loss


# ======================
# 3️⃣ Entraînement
# ======================

def train_vade(X, y=None, latent_dim=10, n_clusters=10, epochs=100, lr=2e-4, verbose=False):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    model = VaDE(input_dim=X.shape[1], latent_dim=latent_dim, n_clusters=n_clusters)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_hat, z, z_mean, z_log_var, gamma = model(X_t)

        # Stabilisation
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        gamma = torch.clamp(gamma, min=1e-10, max=1.0)

        loss = vade_loss(X_t, x_hat, z, z_mean, z_log_var, gamma, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    # Final clustering
    with torch.no_grad():
        _, z, _, _, gamma = model(X_t)
        Z = z.numpy()
        y_pred = np.argmax(gamma.numpy(), axis=1)

    # --- EVITER L'ERREUR ---
    if len(np.unique(y_pred)) < 2:
        sil = 0
    else:
        sil = silhouette_score(Z, y_pred)

    return Z, y_pred, sil

def Vade(X, y=None, n_clusters_low=2, n_clusters_up=9, latent_dim=10, epochs=100):

    best_sil = -1
    best_n = None
    best_Z = None
    best_y_pred = None

    # Tester de n_clusters_low à n_clusters_up INCLUS
    for n_clusters in range(n_clusters_low, n_clusters_up + 1):

        Z, y_pred, sil = train_vade(
            X, y=None,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            epochs=epochs,
            lr=1e-3
        )

        # Détection clusters trop petits
        unique, counts = np.unique(y_pred, return_counts=True)
        threshold = max(20, 0.1 * len(y_pred))
        large_clusters = [k for k, v in zip(unique, counts) if v >= threshold]

        # Critère d'acceptation
        if sil > best_sil and len(large_clusters) >= 2:
            best_sil = sil
            best_n = n_clusters
            best_Z = Z
            best_y_pred = y_pred

    # Cas dégénéré : aucun clustering acceptable
    if best_n is None:
        best_y_pred = np.zeros(len(X), dtype=int)
        best_Z = np.zeros((len(X), latent_dim))

    return best_Z, best_y_pred

# ======================
# 4️⃣ Exemple d’utilisation
# ======================
'''
if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target

    Z, y_pred = train_vade(X, y, latent_dim=5, n_clusters=3, epochs=100)
'''