# depict_fixed.py
"""
DEPICT — Implémentation fidèle + robustifications
Corrections:
 - Instrumentation des phases (pretrain, refine)
 - Protection contre NaN/Inf
 - KMeans safe wrapper + fallback
 - Limitation threads (OMP/BLAS) pour stabilité Windows
 - num_workers=0 par défaut (compatible Windows)
"""

import os
# Limit multi-threading used by numpy/scipy/sklearn to avoid Windows crashes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
# Utils
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

def cluster_acc(y_true, y_pred):
    """ Hungarian algorithm for ACC """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[p, t] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return sum(w[r, c] for r, c in zip(row, col)) / len(y_true)

# ============================================================
# DEPICT Model
# ============================================================

class DEPICT(nn.Module):
    def __init__(self, k=10, latent_dim=32):
        super().__init__()
        self.k = k

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, latent_dim)
        )

        # clustering head
        self.clust_head = nn.Linear(latent_dim, k)

        # ----- Decoder -----
        self.decoder_fc = nn.Linear(latent_dim, 64*7*7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        q = F.softmax(self.clust_head(z), dim=1)
        return q, x_rec, z

# ============================================================
# Target distribution
# ============================================================

def depict_target_distribution(w):
    """
    w : Tensor shape [batch_size, num_clusters]
    Retourne q de même shape normalisée comme DEPICT.
    """
    # fréquence par cluster : [K]
    f = w.sum(dim=0) + 1e-12
    
    # p^2 / f_k : shape [B, K]
    q = (w**2) / f
    
    # normalisation ligne par ligne (sur les clusters)
    q = q / (q.sum(dim=1, keepdim=True) + 1e-12)
    
    return q.detach()

# ============================================================
# Pretraining autoencoder (instrumented)
# ============================================================

def pretrain_autoencoder(model, loader, epochs=40, lr=1e-3, device="cpu", verbose=1):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        n = 0
        grad_norm = 0.0
        for x, _ in loader:
            x = x.to(device)
            _, x_rec, _ = model(x)
            loss = F.mse_loss(x_rec, x, reduction='mean')

            opt.zero_grad()
            loss.backward()

            # grad norm for debug
            gnorm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gnorm += float(p.grad.detach().norm().item()**2)
            grad_norm = gnorm**0.5

            opt.step()

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        mean_loss = total / (n if n>0 else 1)
        if verbose:
            print(f"[PRETRAIN] Epoch {ep}/{epochs} | recon={mean_loss:.6e} | grad_norm={grad_norm:.6e}")

        if mean_loss < 1e-12:
            print("[PRETRAIN] tiny loss -> early stop")
            break

    return model

# ============================================================
# Refinement (DEPICT loss) instrumented & robust
# ============================================================

def refine_depict(model, loader, epochs=60, lr=1e-4, device="cpu", verbose=1):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        n = 0
        any_step = False
        epoch_skipped_batches = 0
        for x, _ in loader:
            x = x.to(device)

            q, x_rec, _ = model(x)
            # clamp q to avoid log(0)
            q = q.clamp(min=1e-8)
            p = depict_target_distribution(q)
            p = p.clamp(min=1e-8)

            # compute losses
            loss_kl = F.kl_div(q.log(), p, reduction="batchmean")
            loss_rec = F.mse_loss(x_rec, x, reduction="mean")
            loss = loss_kl + loss_rec

            # diagnostics: check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                epoch_skipped_batches += 1
                print("[REFINE] Encountered NaN/Inf loss -> skipping batch")
                continue

            opt.zero_grad()
            loss.backward()

            # grad norm
            gnorm = 0.0
            for p_ in model.parameters():
                if p_.grad is not None:
                    gnorm += float(p_.grad.detach().norm().item()**2)
            grad_norm = gnorm**0.5

            # step
            opt.step()
            any_step = True

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        mean_loss = total / (n if n>0 else 1)
        if verbose:
            print(f"[REFINE] Epoch {ep}/{epochs} | loss={mean_loss:.6e} | any_step={any_step} | skipped_batches={epoch_skipped_batches} | grad_norm={grad_norm:.6e}")

        # safety checks
        if not any_step:
            print("[REFINE] No parameter updates this epoch -> early stop")
            break
        if mean_loss < 1e-12:
            print("[REFINE] tiny loss -> early stop")
            break

    return model

# ============================================================
# KMeans safe wrapper
# ============================================================

def cluster_with_kmeans_safe(model, loader_or_dataset, device, batch_size=256):
    """
    Robust KMeans wrapper:
     - accepts DataLoader or Dataset
     - stabilizes Z (no NaN/Inf)
     - casts to float64
     - tries sklearn KMeans then scipy fallback
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.cluster.vq import kmeans2

    # create loader if dataset passed
    if isinstance(loader_or_dataset, Dataset):
        loader = DataLoader(loader_or_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        loader = loader_or_dataset

    model.eval()
    Z_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            # unpack batch robustly
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    x_part, y = batch
                    if isinstance(x_part, (tuple, list)):
                        x = x_part[0]
                    else:
                        x = x_part
                    labels_list.append(y.cpu().numpy())
                else:
                    x = batch[0]
            else:
                x = batch

            x = x.to(device)
            z = model.encode(x).detach().cpu()

            # normalize per-row safely
            z_norm = z.norm(dim=1, keepdim=True)
            z_norm[z_norm == 0] = 1e-8
            z = z / z_norm

            Z_list.append(z)

    if len(Z_list) == 0:
        raise RuntimeError("No embeddings collected (empty loader).")

    Z = torch.cat(Z_list, dim=0).numpy().astype(np.float64)
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)

    labels = np.concatenate(labels_list) if len(labels_list) else None

    # diagnostics
    print("[KMEANS_SAFE] Z.shape", Z.shape, " dtype", Z.dtype)
    print("[KMEANS_SAFE] any NaN:", np.isnan(Z).any(), " any Inf:", np.isinf(Z).any(),
          " min/max:", np.min(Z), np.max(Z))

    # attempt sklearn KMeans (single-thread behaviour controlled via env vars above)
    try:
        with ignore_warnings(category=ConvergenceWarning):
            km = KMeans(n_clusters=model.k, n_init=20, random_state=42)
            preds = km.fit_predict(Z)
        return preds, labels, Z
    except Exception as e:
        print("[KMEANS_SAFE] sklearn KMeans failed with:", repr(e))
        print("[KMEANS_SAFE] Trying scipy kmeans2 fallback.")
        try:
            centroids, labels_k = kmeans2(Z, model.k, minit='points')
            preds = labels_k
            return preds, labels, Z
        except Exception as e2:
            print("[KMEANS_SAFE] fallback kmeans2 also failed:", repr(e2))
            raise RuntimeError("Both KMeans and fallback failed; see logs above.") from e2

# ============================================================
# t-SNE visualisation
# ============================================================

def visualize_tsne(emb, true_labels=None, pred_labels=None, n=2000):
    idx = np.random.choice(len(emb), min(n, len(emb)), replace=False)
    E = emb[idx]

    ts = TSNE(n_components=2, init="pca", random_state=42).fit_transform(E)

    if true_labels is not None:
        plt.figure(figsize=(7,6))
        plt.scatter(ts[:,0], ts[:,1], c=true_labels[idx], cmap='tab10', s=5)
        plt.title("t-SNE – true labels")
        plt.show()

    if pred_labels is not None:
        plt.figure(figsize=(7,6))
        plt.scatter(ts[:,0], ts[:,1], c=pred_labels[idx], cmap='tab10', s=5)
        plt.title("t-SNE – predicted clusters")
        plt.show()

# ============================================================
# Main
# ============================================================

def maindepictminst(args):
    # Make device a torch.device
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("DEVICE:", device)

    # Dataset
    tf = transforms.Compose([transforms.ToTensor()])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)

    if args.n_samples is not None:
        idx = np.random.choice(len(full), args.n_samples, replace=False)
        ds = Subset(full, idx)
    else:
        ds = full

    # DataLoader (num_workers=0 for Windows safety)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model
    model = DEPICT(k=args.k, latent_dim=args.latent_dim).to(device)

    # Pretrain AE
    print("\n===== PRETRAINING AUTOENCODER =====")
    model = pretrain_autoencoder(model, loader,
                                 epochs=args.pretrain_epochs,
                                 lr=args.pretrain_lr,
                                 device=device,
                                 verbose=1)

    # Refinement
    print("\n===== DEPICT CLUSTERING REFINEMENT =====")
    model = refine_depict(model, loader,
                          epochs=args.refine_epochs,
                          lr=args.refine_lr,
                          device=device,
                          verbose=1)

    # Final KMeans (safe)
    print("\n===== FINAL KMEANS =====")
    preds, labels, Z = cluster_with_kmeans_safe(model, loader, device, batch_size=args.batch_size)

    # Metrics
    if labels is not None:
        acc = cluster_acc(labels, preds)
        nmi = NMI(labels, preds)
        ari = ARI(labels, preds)
    else:
        acc = nmi = ari = float("nan")

    sil = silhouette_score(Z, preds) if len(np.unique(preds)) > 1 else float("nan")

    print("\n===== FINAL METRICS =====")
    print(f"ACC = {acc:.4f}")
    print(f"NMI = {nmi:.4f}")
    print(f"ARI = {ari:.4f}") 
    print(f"Silhouette = {sil:.4f}")

    # t-SNE
    visualize_tsne(Z, labels, preds, n=args.tsne_samples)

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "depict_final.pth"))
    print("Model saved to", args.save_dir)

def launchdepictmnist():
    script_dir = Path(__file__).parent.resolve()  # dossier du script
    default_data_dir = script_dir / "data"
    default_data_save = script_dir / "depict_ckpt"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--refine_epochs", type=int, default=40)
    parser.add_argument("--refine_lr", type=float, default=1e-4)
    parser.add_argument("--tsne_samples", type=int, default=2000)
    parser.add_argument("--save_dir", type=str, default=str(default_data_save))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    maindepictminst(args)


