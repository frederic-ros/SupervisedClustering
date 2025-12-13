# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 07:30:08 2025

@author: frederic.ros
"""

# vade_full.py
# Prêt à copier-coller
# Auteur adapt: ChatGPT (pour F. Ros)
# Usage: from vade_full import Vade ; Z, y_pred = Vade(data, ...)

import numpy as np
import copy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---------------------
# VaDE model & loss
# ---------------------
class VaDE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, n_clusters=10):
        super(VaDE, self).__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # encoder: produce hidden then mean/logvar
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(32, latent_dim)
        self.z_log_var = nn.Linear(32, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        # GMM params in latent
        self.pi = nn.Parameter(torch.ones(n_clusters, dtype=torch.float32) / n_clusters)
        self.mu = nn.Parameter(torch.randn(n_clusters, latent_dim, dtype=torch.float32))
        self.log_var = nn.Parameter(torch.zeros(n_clusters, latent_dim, dtype=torch.float32))

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
        # z: (N, D)
        # returns gamma: (N, K)
        # compute log p(z|k) + log pi_k
        z_expand = z.unsqueeze(1)                   # (N,1,D)
        mu_expand = self.mu.unsqueeze(0)            # (1,K,D)
        log_var_expand = self.log_var.unsqueeze(0)  # (1,K,D)
        # compute log probability of z under each Gaussian (diagonal cov)
        # -0.5 * [ sum( (z-mu)^2 / var + log_var ) ]
        inv_var = torch.exp(-log_var_expand)
        term = -0.5 * torch.sum(inv_var * (z_expand - mu_expand) ** 2 + log_var_expand, dim=2)  # (N,K)
        term = term + torch.log(self.pi + 1e-12)  # add log pi
        gamma = torch.softmax(term, dim=1)
        return gamma

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decode(z)
        gamma = self.compute_gamma(z)
        return x_hat, z, z_mean, z_log_var, gamma


def vade_loss(x, x_hat, z, z_mean, z_log_var, gamma, model):
    """
    Loss similar to VaDE paper:
    recon_loss + E_q[ log q(z|x) - log p(z|c) - log p(c) ]
    Implementation: mean MSE + cluster-weighted KL terms.
    """
    # reconstruction (MSE)
    recon_loss = torch.mean((x - x_hat) ** 2)

    # compute expected log p(z|c) under gamma
    z_expand = z.unsqueeze(1)          # (N,1,D)
    mu_expand = model.mu.unsqueeze(0)  # (1,K,D)
    log_var_expand = model.log_var.unsqueeze(0)  # (1,K,D)

    inv_var = torch.exp(-log_var_expand)
    term_pc = 0.5 * torch.sum(log_var_expand + inv_var * (z_expand - mu_expand) ** 2, dim=2)  # (N,K)
    # E_q[ log p(z|c) ] = - sum_k gamma * term_pc  (up to constants)
    p_c_z = torch.sum(gamma * term_pc, dim=1)  # (N,)

    # entropy term from q(z|x): -E_q log q = -0.5 * sum(1 + log_var) per point
    q_entropy = -0.5 * torch.sum(1 + z_log_var, dim=1)

    # KL-like per-sample (we combine signs to get positive loss)
    kl_loss = torch.mean(p_c_z - q_entropy - torch.sum(gamma * torch.log(model.pi + 1e-12), dim=1))

    return recon_loss + kl_loss


# ---------------------
# Small AE for pretraining
# ---------------------
class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(SimpleAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def pretrain_ae(X_scaled, input_dim, latent_dim=10, pretrain_epochs=100, lr=1e-3, verbose=False):
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    ae = SimpleAE(input_dim=input_dim, latent_dim=latent_dim)
    opt = optim.Adam(ae.parameters(), lr=lr)
    for ep in range(pretrain_epochs):
        ae.train()
        opt.zero_grad()
        x_hat, z = ae(X_t)
        loss = F.mse_loss(x_hat, X_t)
        loss.backward()
        opt.step()
        if verbose and ((ep+1) % 50 == 0 or ep == 0):
            print(f"[AE pretrain {ep+1}/{pretrain_epochs}] loss={loss.item():.6f}")
    ae.eval()
    with torch.no_grad():
        z0 = ae.encode(torch.tensor(X_scaled, dtype=torch.float32)).cpu().numpy()
    return ae, z0


# ---------------------
# helper: copy compatible weights from AE to VaDE encoder (best-effort)
# ---------------------
def copy_encoder_weights_from_ae(ae, vade_model):
    """
    Best-effort copy: match linear layers weights by size.
    """
    ae_state = ae.state_dict()
    vade_state = vade_model.state_dict()
    new_state = {}
    for k, v in ae_state.items():
        # try to find a vade_state key with same shape
        for vk, vv in vade_state.items():
            if vv.shape == v.shape and vk not in new_state:
                new_state[vk] = v.clone()
                break
    # update vade_state
    vade_state.update(new_state)
    vade_model.load_state_dict(vade_state)


# ---------------------
# train_vade (fine-tune VaDE initialized with GMM+AE)
# ---------------------
def train_vade(X, y=None, latent_dim=10, n_clusters=10, epochs=100, lr=2e-4,
               init_gmm=None, init_ae=None, verbose=False, device='cpu'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model = VaDE(input_dim=X.shape[1], latent_dim=latent_dim, n_clusters=n_clusters).to(device)

    # copy AE encoder weights if possible (best-effort)
    if init_ae is not None:
        try:
            copy_encoder_weights_from_ae(init_ae, model)
        except Exception:
            pass

    # initialize GMM params if provided
    if init_gmm is not None:
        with torch.no_grad():
            model.pi.data = torch.tensor(init_gmm.weights_, dtype=torch.float32).to(device)
            model.mu.data = torch.tensor(init_gmm.means_, dtype=torch.float32).to(device)
            cov = init_gmm.covariances_
            if cov.ndim == 3:
                diag = np.array([np.diag(cov[k]) for k in range(cov.shape[0])])
            else:
                diag = cov
            diag = np.maximum(diag, 1e-6)
            model.log_var.data = torch.tensor(np.log(diag), dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, z, z_mean, z_log_var, gamma = model(X_t)

        # numerical stabilization (use clamp on forwarded tensors)
        z_log_var = torch.clamp(z_log_var, min=-10.0, max=10.0)
        gamma = torch.clamp(gamma, min=1e-8, max=1.0)

        loss = vade_loss(X_t, x_hat, z, z_mean, z_log_var, gamma, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if verbose and ((epoch+1) % 50 == 0 or epoch == 0):
            print(f"[VaDE ft {epoch+1}/{epochs}] loss={loss.item():.6f}")

    # final deterministic embedding and labels (use z_mean for stability)
    model.eval()
    with torch.no_grad():
        X_t_cpu = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        z_mean_t, z_log_var_t = model.encode(X_t_cpu)
        # compute gamma on z_mean
        gamma_t = model.compute_gamma(z_mean_t)
        gamma_np = gamma_t.cpu().numpy()
        y_pred = np.argmax(gamma_np, axis=1)
        Z = z_mean_t.cpu().numpy()

    if len(np.unique(y_pred)) < 2:
        sil = 0.0
    else:
        sil = silhouette_score(Z, y_pred)

    return Z, y_pred, sil

# ---------------------
# Vade orchestration: pretrain AE once, loop n_clusters with GMM init
# ---------------------

def Vade(X, y=None, n_clusters_low=2, n_clusters_up=9, latent_dim=10, epochs=100,
         pretrain_epochs=100, ae_lr=1e-3, vade_lr=2e-4, verbose=False, device='cpu'):
    """
    Returns best_Z, best_y_pred according to silhouette (protocol unchanged).
    Robust version: safe automatic exploration of n_clusters.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pretrain AE one time
    ae, Z0 = pretrain_ae(
        X_scaled,
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        pretrain_epochs=pretrain_epochs,
        lr=ae_lr,
        verbose=verbose
    )

    best_sil = -1.0
    best_n = None
    best_Z = None
    best_y_pred = None

    for n_clusters in range(n_clusters_low, n_clusters_up + 1):

        # -------------------------
        # Robust GMM initialization
        # -------------------------
        try:
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='diag',
                reg_covar=1e-4,   # prevents ill-defined covariance
                n_init=5,         # avoids degenerate EM solutions
                random_state=42
            )
            gmm.fit(Z0)

            # Optional but recommended: skip degenerate effective clustering
            labels0 = gmm.predict(Z0)
            if len(np.unique(labels0)) < 2:
                if verbose:
                    print(f"[Skip k={n_clusters}] single effective cluster")
                continue

        except ValueError as e:
            if verbose:
                print(f"[Skip k={n_clusters}] GMM failed: {e}")
            continue

        # -------------------------
        # Fine-tune VaDE
        # -------------------------
        Z, y_pred, sil = train_vade(
            X,
            y=None,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            epochs=epochs,
            lr=vade_lr,
            init_gmm=gmm,
            init_ae=ae,
            verbose=verbose,
            device=device
        )

        unique, counts = np.unique(y_pred, return_counts=True)
        threshold = max(5, int(0.05 * len(y_pred)))
        large_clusters = [k for k, v in zip(unique, counts) if v >= threshold]

        if sil > best_sil and len(large_clusters) >= 2:
            best_sil = sil
            best_n = n_clusters
            best_Z = Z
            best_y_pred = y_pred
            if verbose:
                print(f"New best: k={best_n}, sil={best_sil:.4f}")

    if best_n is None:
        if verbose:
            print("No valid clustering found, returning zeros.")
        best_y_pred = np.zeros(len(X), dtype=int)
        best_Z = np.zeros((len(X), latent_dim))
    else:
        if verbose:
            print(f"Selected best_n={best_n} (sil={best_sil:.4f})")

    return best_Z, best_y_pred


'''
# ---------------------
# Quick test (Iris) when running as script
# ---------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris().data
    Z, y_pred = Vade(data, n_clusters_low=2, n_clusters_up=5, latent_dim=3,
                     epochs=80, pretrain_epochs=80, verbose=True)
    print("Unique labels:", np.unique(y_pred))
'''