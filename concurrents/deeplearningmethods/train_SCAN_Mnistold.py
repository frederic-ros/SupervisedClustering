# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:18:15 2025

@author: frederic.ros
"""
# -*- coding: utf-8 -*-
"""
SCAN clustering sur MNIST/KMNIST
Auteur: Réécriture complète
Évaluation directement sur les sorties probabilistes (softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# 1. Paramètres
# -----------------------------
dataset_name = 'MNIST'       # 'MNIST' ou 'KMNIST'
batch_size = 128
epochs = 100
num_clusters = 10
embedding_dim = 32
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pour tests rapides
n_samples_per_class = 5000      # nombre d'images par classe pour échantillon équilibré

# -----------------------------
# 2. Transformations et augmentations
# -----------------------------
augmentation = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
to_pil = ToPILImage()

# -----------------------------
# 3. Charger dataset complet
# -----------------------------
if dataset_name == 'MNIST':
    dataset_full = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
elif dataset_name == 'KMNIST':
    dataset_full = datasets.KMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.KMNIST('./data', train=False, download=True, transform=transforms.ToTensor())
else:
    raise ValueError("Dataset inconnu")

# -----------------------------
# 4. Fonction pour subset équilibré par classe
# -----------------------------
def balanced_subset(dataset, n_per_class, seed=42):
    labels = np.array(dataset.targets)
    indices = []
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        class_idx = np.where(labels==c)[0]
        if len(class_idx) >= n_per_class:
            chosen = rng.choice(class_idx, n_per_class, replace=False)
        else:
            chosen = rng.choice(class_idx, n_per_class, replace=True)
        indices.extend(chosen)
    return Subset(dataset, indices)

train_dataset = balanced_subset(dataset_full, n_samples_per_class)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Nombre d'images chargées pour l'entraînement : {len(train_dataset)}")

# -----------------------------
# 5. Backbone CNN
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
        self.fc = nn.Linear(128*7*7, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

# -----------------------------
# 6. SCAN Head
# -----------------------------
class SCANHead(nn.Module):
    def __init__(self, embedding_dim, num_clusters):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_clusters)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)

# -----------------------------
# 7. SCAN Loss
# -----------------------------
def scan_loss(outputs, outputs_aug):
    consistency = -torch.sum(outputs * torch.log(outputs_aug + 1e-6), dim=1).mean()
    p = torch.mean(outputs, dim=0)
    entropy = torch.sum(p * torch.log(p + 1e-6))
    return consistency + entropy

# -----------------------------
# 8. Modèle + optimiseur
# -----------------------------
backbone = CNNBackbone(embedding_dim).to(device)
head = SCANHead(embedding_dim, num_clusters).to(device)
optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=learning_rate)

# -----------------------------
# 9. Training Loop
# -----------------------------
for epoch in range(epochs):
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
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_dataset):.4f}")

# -----------------------------
# 10. Extraire embeddings et probabilités softmax
# -----------------------------
def extract_embeddings_probs(loader, backbone, head):
    backbone.eval()
    head.eval()
    embeddings, probs, labels_all = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = backbone(imgs)
            out = head(emb)
            embeddings.append(emb.cpu().numpy())
            probs.append(out.cpu().numpy())
            labels_all.append(labels.numpy())
    embeddings = np.vstack(embeddings)
    probs = np.vstack(probs)
    labels_all = np.hstack(labels_all)
    return embeddings, probs, labels_all

train_embeddings, train_probs, train_labels = extract_embeddings_probs(train_loader, backbone, head)
test_embeddings, test_probs, test_labels = extract_embeddings_probs(test_loader, backbone, head)

# -----------------------------
# 11. Prédictions SCAN (probabilités)
# -----------------------------
train_preds = np.argmax(train_probs, axis=1)
test_preds = np.argmax(test_probs, axis=1)

# -----------------------------
# 12. Visualisation t-SNE
# -----------------------------
def visualize_latent_space(embeddings, labels, preds=None, n_samples=1000, title="t-SNE"):
    if embeddings.shape[0] > n_samples:
        idx = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        if preds is not None:
            preds = preds[idx]
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6,6))
    if preds is None:
        sc = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap='tab10', s=5)
        plt.title(f"{title} - true labels")
    else:
        sc = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=preds, cmap='tab10', s=5)
        plt.title(f"{title} - predicted clusters")
    plt.colorbar(sc)
    plt.show()

visualize_latent_space(train_embeddings, train_labels, preds=train_preds, n_samples=1000, title="Train embeddings SCAN")

# -----------------------------
# 13. Évaluation clustering
# -----------------------------
def evaluate_clustering(labels_true, labels_pred, embeddings):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    sil = silhouette_score(embeddings, labels_pred) if len(np.unique(labels_pred)) > 1 else np.nan
    return nmi, ari, sil

nmi_train, ari_train, sil_train = evaluate_clustering(train_labels, train_preds, train_embeddings)
nmi_test, ari_test, sil_test = evaluate_clustering(test_labels, test_preds, test_embeddings)

print("Clustering metrics - Train set:")
print(f"  NMI: {nmi_train:.4f}, ARI: {ari_train:.4f}, Silhouette: {sil_train:.4f}")
print("Clustering metrics - Test set:")
print(f"  NMI: {nmi_test:.4f}, ARI: {ari_test:.4f}, Silhouette: {sil_test:.4f}")
