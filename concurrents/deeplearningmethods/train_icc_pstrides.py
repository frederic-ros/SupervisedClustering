#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=TqdmWarning)

import os, json, argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Dataset
# ----------------------------
class TwoCropsTransform:
    def __init__(self, transform): self.transform = transform
    def __call__(self, x): return self.transform(x), self.transform(x)

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
        self.data = torch.tensor(inputs[indices], dtype=torch.uint8)
        self.targets = torch.tensor(labels[indices], dtype=torch.long)
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform: img = self.transform(img)
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

# ----------------------------
# Model
# ----------------------------
class ResBackbone(nn.Module):
    def __init__(self, in_ch=1, pretrained=False):
        super().__init__()
        rn = models.resnet18(pretrained=pretrained)
        if in_ch != 3:
            rn.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.body = nn.Sequential(*list(rn.children())[:-1])
        self.out_dim = 512
    def forward(self, x):
        f = self.body(x)
        return f.view(f.size(0), -1)

class IICModelStable(nn.Module):
    def __init__(self, in_ch=1, num_clusters=3, feat_dim=32, proj_hidden=256, over_factors=(1,3), use_pretrained=False):
        super().__init__()
        self.back = ResBackbone(in_ch, pretrained=use_pretrained)
        self.proj = nn.Sequential(
            nn.Linear(self.back.out_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, feat_dim)
        )
        self.heads = nn.ModuleList()
        for f in over_factors:
            k = int(num_clusters*f)
            self.heads.append(nn.Linear(feat_dim, k))
    def forward(self, x):
        f = self.back(x)
        z = self.proj(f)
        probs = [F.softmax(h(z), dim=1) for h in self.heads]
        return probs, z

# ----------------------------
# IIC loss
# ----------------------------
def iic_loss(p1_list, p2_list, eps=1e-10, lambda_entropy=0.5):
    losses, regs = [], []
    for p1,p2 in zip(p1_list,p2_list):
        P = torch.matmul(p1.t(), p2)
        P = P / (P.sum()+eps)
        pi = P.sum(dim=1).view(-1,1)
        pj = P.sum(dim=0).view(1,-1)
        mi = (P*(torch.log(P+eps)-torch.log(pi+eps)-torch.log(pj+eps))).sum()
        losses.append(-mi)
        marginal = 0.5*(p1.mean(dim=0)+p2.mean(dim=0))
        regs.append((marginal*torch.log(marginal+eps)).sum())
    return torch.stack(losses).mean() + lambda_entropy*torch.stack(regs).mean()

# ----------------------------
# Training
# ----------------------------
def train_iic(model, dataset, epochs=50, batch_size=128, device='cpu',
              lr=3e-4, weight_decay=1e-5, lambda_entropy=0.5,
              num_workers=0, eval_every=1, verbose=True, batch_log=5):

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA non disponible, utilisation du CPU")
        device = 'cpu'

    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        running_loss = 0
        for i, (xb, _) in enumerate(loader, 1):
            x1, x2 = xb
            x1 = x1.to(device)
            x2 = x2.to(device)
            p1_list, _ = model(x1)
            p2_list, _ = model(x2)
            loss = iic_loss(p1_list, p2_list, lambda_entropy=lambda_entropy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()*x1.size(0)
            running_loss += loss.item()
            if verbose and i % batch_log == 0:
                print(f"Epoch {ep:03d}, Batch {i}/{len(loader)} - Avg Batch Loss: {running_loss/batch_log:.4f}")
                running_loss = 0
        avg_loss = total_loss / len(dataset)
        if verbose and (ep % eval_every == 0 or ep==epochs):
            print(f"Epoch {ep:03d}/{epochs} - Avg Epoch Loss: {avg_loss:.4f}")
    return model

# ----------------------------
# Metrics
# ----------------------------
def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1
    ind_row, ind_col = linear_sum_assignment(w.max()-w)
    return sum([w[r,c] for r,c in zip(ind_row, ind_col)])/len(y_true)

def evaluate_model(model, dataset, n_samples_eval=None, head_idx=0, device='cpu'):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    feats, labels_list, preds_list = [], [], []
    count = 0
    with torch.no_grad():
        for xb, y in loader:
            if isinstance(xb,(tuple,list)): x1,_ = xb
            else: x1 = xb
            x1 = x1.to(device)
            p_list, z = model(x1)
            feats.append(z.cpu().numpy())
            preds_list.append(p_list[head_idx].argmax(dim=1).cpu().numpy())
            labels_list.append(y.numpy())
            count += x1.size(0)
            if n_samples_eval and count>=n_samples_eval: break
    feats = np.concatenate(feats)
    preds = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    if n_samples_eval: feats, preds, labels = feats[:n_samples_eval], preds[:n_samples_eval], labels[:n_samples_eval]
    acc = cluster_accuracy(labels, preds)
    nmi = NMI(labels, preds)
    ari = ARI(labels, preds)
    metrics = {'ACC':acc, 'NMI':nmi, 'ARI':ari}
    return metrics, preds, labels

# ----------------------------
# Embeddings & t-SNE
# ----------------------------
def extract_embeddings(model, dataset, device='cpu', max_samples=None):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    feats, labels_list = [], []
    count = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for xb, y in loader:
            if isinstance(xb,(tuple,list)): x1,_ = xb
            else: x1 = xb
            x1 = x1.to(device)
            _, z = model(x1)
            feats.append(z.cpu().numpy())
            labels_list.append(y.numpy())
            count += x1.size(0)
            if max_samples and count >= max_samples: break
    feats = np.concatenate(feats)
    labels = np.concatenate(labels_list)
    if max_samples: feats, labels = feats[:max_samples], labels[:max_samples]
    return feats, labels

   
def visualize_tsne_from_embeddings(feats, labels, title="t-SNE latent space"):
    tsne = TSNE(n_components=2, random_state=42, init='pca', n_jobs=-1)
    tsne_feats = tsne.fit_transform(feats)

    plt.figure(figsize=(6,6))

    # Couleurs personnalisées pour 3 classes
    colors = ['red', 'green', 'blue']
    class_names = np.unique(labels)

    for i, cls in enumerate(class_names):
        idx = labels == cls
        plt.scatter(tsne_feats[idx, 0], tsne_feats[idx, 1],
                    c=colors[i], label=f'Classe {cls}', s=5)

    plt.title(title)
    plt.legend(title="Classes")
    plt.show()

  
# ----------------------------
# Main
# ----------------------------
def parse_args_main():
    script_dir = Path(__file__).parent.resolve()  # dossier du script
    default_data_dir = script_dir / "data"
    
    parser = argparse.ArgumentParser(description="IIC dSprites - All-in-one")
    #parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir))
    parser.add_argument("--target_latent", type=str, default="shape")
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--per_class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--lambda_entropy", type=float, default=5.0)
    parser.add_argument("--feat_dim", type=int, default=100)
    parser.add_argument("--proj_hidden", type=int, default=256)
    parser.add_argument("--over_factors", nargs="+", type=int, default=[1,3])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--tsne", type=bool, default=True)
    parser.add_argument("--eval_samples", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_iic")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()

# Extraction des embeddings et labels correctement
def extract_embeddings_with_true_labels(model, dataset, device='cpu'):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    feats, labels_list = [], []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for xb, y in loader:

            # Cas TwoCropsTransform : xb = (view1, view2)
            if isinstance(xb, (tuple, list)):
                x1, _ = xb
            else:
                x1 = xb

            x1 = x1.to(device)
            _, z = model(x1)  # embeddings

            feats.append(z.cpu().numpy())
            labels_list.append(y.numpy())  # vrais labels du dataset

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print("Labels extraits uniques :", np.unique(labels))  # DEBUG IMPORTANT
    print("Nombre total d'embeddings :", len(feats))

    return feats, labels

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def compute_metrics(feats, true_labels, num_clusters, seed=0):
    # Clustering sur les embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(feats)
    pred_labels = kmeans.labels_

    # --- Metrics supervisées ---
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    # --- CVI sur les embeddings ---
    silhouette = silhouette_score(feats, pred_labels)
    ch = calinski_harabasz_score(feats, pred_labels)
    db = davies_bouldin_score(feats, pred_labels)

    metrics = {
        "ARI": ari,
        "NMI": nmi,
        "Silhouette": silhouette,
        "Calinski_Harabasz": ch,
        "Davies_Bouldin": db
    }
    return metrics, pred_labels


from sklearn.metrics import silhouette_score

def mainiccdsprites():
    args = parse_args_main()

    # Dossier de sauvegarde
    if args.run_name is None:
        args.run_name = datetime.now().strftime("IIC_%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # Transforms
    augment = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
        transforms.RandomAffine(degrees=30, translate=(0.15,0.15), scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    two_crops = TwoCropsTransform(augment)

    # Dataset d'entraînement
    ds_train = DSprites(args.data_dir, transform=two_crops, 
                        n_samples=args.n_samples,
                        target_latent=args.target_latent, 
                        seed=SEED)

    if args.per_class:
        idx_bal = balanced_subset_indices(ds_train, args.per_class, seed=SEED)
        ds_train = Subset(ds_train, idx_bal)

    # Vérification équilibre des classes
    labels_train_full = np.array([y for _, y in ds_train])
    unique, counts = np.unique(labels_train_full, return_counts=True)
    print("\nRépartition des classes dans le dataset d'entraînement :")
    for u, c in zip(unique, counts):
        print(f"  Classe {u} : {c} exemples")

    # Modèle
    num_clusters = len(unique)
    model = IICModelStable(in_ch=1, num_clusters=num_clusters,
                           feat_dim=args.feat_dim,
                           proj_hidden=args.proj_hidden,
                           over_factors=tuple(args.over_factors))

    device = 'cpu'
    print(f"\nTraining on device: {device}\n")

    # Entraînement
    if not args.eval_only:
        model = train_iic(model, ds_train, 
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr=args.lr,
                          weight_decay=args.wd,
                          lambda_entropy=args.lambda_entropy,
                          device=device)

    # Extraction des embeddings et labels → TOUT le dataset
    feats_train, labels_train = extract_embeddings_with_true_labels(
        model, ds_train, device=device
    )
    # --- Metrics ---
    metrics, preds = compute_metrics(
        feats_train, 
        labels_train, 
        num_clusters=num_clusters, 
        seed=SEED
        )

    print("\n===== METRICS ON TRAINING EMBEDDINGS =====")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

        print("Vérification labels t-SNE :", np.unique(labels_train))

    # t-SNE
    visualize_tsne_from_embeddings(
        feats_train, labels_train,
        title="t-SNE latent space - Training set"
    )

#.........................................................................................
def launchiccdsprites():
    mainiccdsprites()


