# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 18:56:09 2025

@author: frederic.ros
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPICT adapted to DSprites (64x64 grayscale)
Save as: depict_dsprites.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[p, t] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return sum(w[r, c] for r, c in zip(row, col)) / len(y_true)

# ---------------------------
# DSprites Dataset
# ---------------------------
class DSprites(Dataset):
    filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    target_names = {"color":0,"shape":1,"scale":2,"orientation":3,"pos_x":4,"pos_y":5}
    def __init__(self, data_dir, transform=None, n_samples=None, target_latent="shape", seed=0):
        data_dir = Path(data_dir)
        npz_file = np.load(data_dir/self.filename)
        inputs, labels = npz_file["imgs"]*255, npz_file["latents_classes"][:, self.target_names[target_latent]]
        rng = np.random.default_rng(seed)
        indices = np.arange(len(labels))
        rng.shuffle(indices)
        if n_samples is not None:
            indices = indices[:n_samples]
        # store as uint8 tensor but we'll convert to PIL in __getitem__
        self.data = torch.tensor(inputs[indices], dtype=torch.uint8)
        self.targets = torch.tensor(labels[indices], dtype=torch.long)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        img = Image.fromarray(img.numpy(), mode="L")  # grayscale PIL image
        if self.transform:
            img = self.transform(img)
        return img, target

def balanced_subset_indices(dataset, per_class, seed=42):
    labels = np.array([y for _, y in dataset])
    selected_indices = []
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        inds = np.where(labels==c)[0]
        rng.shuffle(inds)
        selected_indices.extend(inds[:per_class])
    return selected_indices

# ---------------------------
# DEPICT model adapted for 64x64
# ---------------------------
class DEPICT(nn.Module):
    def __init__(self, k=3, latent_dim=32):
        super().__init__()
        self.k = k
        # Encoder: 64x64 -> /2 -> 32x32 -> /2 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 16x16
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64*16*16, latent_dim)
        )
        self.clust_head = nn.Linear(latent_dim, k)
        # Decoder: latent -> 64*16*16 -> transpose convs to 64x64
        self.decoder_fc = nn.Linear(latent_dim, 64*16*16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 64, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        q = F.softmax(self.clust_head(z), dim=1)
        return q, x_rec, z

# ---------------------------
# DEPICT target distribution
# ---------------------------
def depict_target_distribution(w):
    # w: (B, K)
    f = w.sum(dim=0) + 1e-12
    q = (w**2) / f
    q = q / (q.sum(dim=1, keepdim=True) + 1e-12)
    return q.detach()

# ---------------------------
# Pretrain & refine functions
# ---------------------------
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

def refine_depict(model, loader, epochs=60, lr=1e-4, device="cpu", verbose=1):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        n = 0
        any_step = False
        epoch_skipped_batches = 0
        grad_norm = 0.0
        for x, _ in loader:
            x = x.to(device)
            q, x_rec, _ = model(x)
            q = q.clamp(min=1e-8)
            p = depict_target_distribution(q).clamp(min=1e-8)
            loss_kl = F.kl_div(q.log(), p, reduction="batchmean")
            loss_rec = F.mse_loss(x_rec, x, reduction="mean")
            loss = loss_kl + loss_rec
            if torch.isnan(loss) or torch.isinf(loss):
                epoch_skipped_batches += 1
                print("[REFINE] NaN/Inf loss -> skipping batch")
                continue
            opt.zero_grad()
            loss.backward()
            # grad norm
            gnorm = 0.0
            for p_ in model.parameters():
                if p_.grad is not None:
                    gnorm += float(p_.grad.detach().norm().item()**2)
            grad_norm = gnorm**0.5
            opt.step()
            any_step = True
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        mean_loss = total / (n if n>0 else 1)
        if verbose:
            print(f"[REFINE] Epoch {ep}/{epochs} | loss={mean_loss:.6e} | any_step={any_step} | skipped={epoch_skipped_batches} | grad_norm={grad_norm:.6e}")
        if not any_step:
            print("[REFINE] No parameter updates this epoch -> early stop")
            break
        if mean_loss < 1e-12:
            print("[REFINE] tiny loss -> early stop")
            break
    return model

# ---------------------------
# Safe KMeans wrapper
# ---------------------------
def cluster_with_kmeans_safe(model, loader_or_dataset, device, batch_size=256):
    import numpy as np
    from scipy.cluster.vq import kmeans2
    # create loader if dataset
    if isinstance(loader_or_dataset, Dataset):
        loader = DataLoader(loader_or_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        loader = loader_or_dataset
    model.eval()
    Z_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
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
            z_norm = z.norm(dim=1, keepdim=True)
            z_norm[z_norm == 0] = 1e-8
            z = z / z_norm
            Z_list.append(z)
    if len(Z_list) == 0:
        raise RuntimeError("No embeddings collected.")
    Z = torch.cat(Z_list, dim=0).numpy().astype(np.float64)
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)
    labels = np.concatenate(labels_list) if len(labels_list) else None
    print("[KMEANS_SAFE] Z.shape", Z.shape, " dtype", Z.dtype)
    try:
        with ignore_warnings(category=ConvergenceWarning):
            km = KMeans(n_clusters=model.k, n_init=20, random_state=42)
            preds = km.fit_predict(Z)
        return preds, labels, Z
    except Exception as e:
        print("[KMEANS_SAFE] sklearn KMeans failed:", repr(e))
        try:
            centroids, labels_k = kmeans2(Z, model.k, minit='points')
            preds = labels_k
            return preds, labels, Z
        except Exception as e2:
            print("[KMEANS_SAFE] fallback failed:", repr(e2))
            raise RuntimeError("KMeans both methods failed.") from e2

# ---------------------------
# t-SNE viz
# ---------------------------
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

# ---------------------------
# Main script
# ---------------------------
def to_python(obj):
    if isinstance(obj, dict):
        return {to_python(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return to_python(obj.cpu().numpy())
    return obj

def parse_args():
    script_dir = Path(__file__).parent.resolve()  # dossier du script
    default_data_dir = script_dir / "data"
    default_data_save = script_dir / "depict_dsprites_ckpt"
    
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(default_data_dir))
    p.add_argument("--target_latent", type=str, default="shape", choices=["color","shape","scale","orientation","pos_x","pos_y"])
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--per_class", type=int, default=1000)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--pretrain_lr", type=float, default=1e-3)
    p.add_argument("--refine_epochs", type=int, default=40)
    p.add_argument("--refine_lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--tsne_samples", type=int, default=2000)
    p.add_argument("--save_dir", type=str, default=str(default_data_save))
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()

def maindepictdsprites(args):
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("DEVICE:", device)
    os.makedirs(args.save_dir, exist_ok=True)
    run_name = datetime.now().strftime("depict_dsprites_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # transforms for DSprites (they are 64x64 already)
    normalize = transforms.Normalize([0.5], [0.5])
    tf = transforms.Compose([transforms.ToTensor(), normalize])

    # dataset
    ds_all = DSprites(Path(args.data_dir), transform=tf, n_samples=args.n_samples, target_latent=args.target_latent, seed=42)
    print("Raw DSprites loaded, total samples:", len(ds_all))

    # balanced subset per class if requested
    if args.per_class:
        idx_bal = balanced_subset_indices(ds_all, per_class=args.per_class, seed=42)
        dataset = Subset(ds_all, idx_bal)
    else:
        dataset = ds_all

    labels_check = np.array([y for _, y in dataset])
    uniques, counts = np.unique(labels_check, return_counts=True)
    print("Label distribution (used):")
    for u,c in zip(uniques, counts):
        print(f"  class {u} : {c}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # model
    model = DEPICT(k=args.k, latent_dim=args.latent_dim).to(device)

    # pretrain
    print("\n=== PRETRAIN AUTOENCODER ===")
    pretrain_autoencoder(model, loader, epochs=args.pretrain_epochs, lr=args.pretrain_lr, device=device, verbose=1)

    # refine
    print("\n=== REFINEMENT (DEPICT) ===")
    refine_depict(model, loader, epochs=args.refine_epochs, lr=args.refine_lr, device=device, verbose=1)

    # final clustering via safe kmeans
    print("\n=== FINAL KMEANS ===")
    preds, labels, Z = cluster_with_kmeans_safe(model, loader, device, batch_size=args.batch_size)

    # metrics
    if labels is not None:
        acc = cluster_acc(labels, preds)
        nmi = NMI(labels, preds)
        ari = ARI(labels, preds)
    else:
        acc = nmi = ari = float("nan")
    sil = silhouette_score(Z, preds) if len(np.unique(preds)) > 1 else float("nan")

    print("\n=== FINAL METRICS ===")
    print(f"ACC = {acc:.4f}")
    print(f"NMI = {nmi:.4f}")
    print(f"ARI = {ari:.4f}")
    print(f"Silhouette = {sil:.4f}")

    # t-SNE
    visualize_tsne(Z, labels, preds, n=args.tsne_samples)

    # save model & metrics (JSON-safe)
    torch.save(model.state_dict(), os.path.join(out_dir, "depict_dsprites_final.pth"))
    out = {
        "acc": float(acc),
        "nmi": float(nmi),
        "ari": float(ari),
        "silhouette": float(sil),
        "label_distribution": {int(u): int(c) for u,c in zip(uniques, counts)}
    }
    with open(os.path.join(out_dir, "metrics_final.json"), "w") as f:
        json.dump(to_python(out), f, indent=2)

    print("Saved outputs in:", out_dir)

def launchdepictdsprites():
    args = parse_args()
    maindepictdsprites(args)
