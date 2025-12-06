# -*- coding: utf-8 -*-
"""
SCAN clustering sur dSprites
Auteur: Réécriture professionnelle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import argparse
from pathlib import Path
# -----------------------------
# DSprites Dataset
# -----------------------------
class DSprites(Dataset):
    filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    target_names = {"color":0,"shape":1,"scale":2,"orientation":3,"pos_x":4,"pos_y":5}

    def __init__(self, data_dir, transform=None, n_samples=None, target_latent="shape", seed=0):
        data_dir = Path(data_dir)
        npz_file = np.load(data_dir/self.filename)
        inputs = npz_file["imgs"]*255
        labels = npz_file["latents_classes"][:, self.target_names[target_latent]]
        rng = np.random.default_rng(seed)
        indices = np.arange(len(labels))
        rng.shuffle(indices)
        if n_samples is not None:
            indices = indices[:n_samples]
        self.data = torch.tensor(inputs[indices], dtype=torch.uint8)
        self.targets = torch.tensor(labels[indices], dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        img = Image.fromarray(img.numpy(), mode="L")
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

# -----------------------------
# Backbone CNN
# -----------------------------
class CNNBackbone(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(128*16*16, embedding_dim)  # dSprites 64x64 -> after 2 pool2d

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, dim=1)

# -----------------------------
# SCAN Head
# -----------------------------
class SCANHead(nn.Module):
    def __init__(self, embedding_dim, num_clusters):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_clusters)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)

# -----------------------------
# SCAN Loss
# -----------------------------
def scan_loss(outputs, outputs_aug):
    consistency = -torch.sum(outputs * torch.log(outputs_aug + 1e-6), dim=1).mean()
    p = torch.mean(outputs, dim=0)
    entropy = torch.sum(p * torch.log(p + 1e-6))
    return consistency + entropy

# -----------------------------
# Extraire embeddings
# -----------------------------
def extract_embeddings(loader, model):
    model.eval()
    embeddings, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = model(imgs).detach().cpu().numpy()
            embeddings.append(emb)
            labels_all.append(labels.numpy())
    return np.vstack(embeddings), np.hstack(labels_all)

# -----------------------------
# Visualisation t-SNE
# -----------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import numpy as np

def visualize_latent_space(embeddings, labels, title="t-SNE", n_samples=1000):

    # Sous-échantillonnage
    if embeddings.shape[0] > n_samples:
        idx = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    emb_2d = tsne.fit_transform(embeddings)

    # Récupère les classes uniques
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Palette de couleurs
    base_cmap = plt.get_cmap('tab10')
    colors = base_cmap.colors[:n_classes]

    plt.figure(figsize=(6, 6))

    # Plot classe par classe → permet la légende propre
    for i, cls in enumerate(unique_classes):
        idx = labels == cls
        plt.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            s=5,
            color=colors[i],
            label=f"classe {cls}"
        )

    plt.title(title)
    plt.legend(markerscale=3, fontsize=10)
    plt.show()


# -----------------------------
# Évaluation clustering
# -----------------------------
def evaluate_clustering(labels_true, probs):
    preds = np.argmax(probs, axis=1)
    nmi = normalized_mutual_info_score(labels_true, preds)
    ari = adjusted_rand_score(labels_true, preds)
    sil = silhouette_score(probs, preds) if len(np.unique(preds))>1 else np.nan
    return nmi, ari, sil

# -----------------------------
# Main
# -----------------------------
def mainscandsprites():
    script_dir = Path(__file__).parent.resolve()  # dossier du script
    default_data_dir = script_dir / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir), help="Chemin vers dSprites npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--n_samples_per_class", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentation = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    to_pil = ToPILImage()

    # Dataset
    dataset_full = DSprites(args.data_dir, transform=transforms.ToTensor(), n_samples=args.n_samples_per_class, target_latent="shape")
    train_loader = DataLoader(dataset_full, batch_size=args.batch_size, shuffle=True)

    # Modèles
    backbone = CNNBackbone(args.embedding_dim).to(device)
    head = SCANHead(args.embedding_dim, args.num_clusters).to(device)
    optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)

    # Training SCAN
    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        total_loss = 0
        for imgs, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            imgs_aug = torch.stack([augmentation(to_pil(img.cpu())) for img in imgs]).to(device)
            emb = backbone(imgs)
            emb_aug = backbone(imgs_aug)
            out = head(emb)
            out_aug = head(emb_aug)
            loss = scan_loss(out, out_aug)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {total_loss/len(dataset_full):.6f}")

    # Embeddings et sorties probabilistes
    backbone.eval()
    head.eval()
    probs_list = []
    labels_all = []
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            emb = backbone(imgs)
            probs = head(emb)
            probs_list.append(probs.cpu().numpy())
            labels_all.append(labels.numpy())
    probs_all = np.vstack(probs_list)
    labels_all = np.hstack(labels_all)

    # Évaluation
    nmi, ari, sil = evaluate_clustering(labels_all, probs_all)
    print("===== FINAL METRICS =====")
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, Silhouette: {sil:.4f}")

    # t-SNE
    visualize_latent_space(probs_all, labels_all, title="SCAN dSprites - Probabilistic Clusters")

def launchscandsprites():
    mainscandsprites()