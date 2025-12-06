# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# iic_mnist_b_latent.py - IIC (paper-style) for MNIST / KMNIST
# Version corrigée: JSON-safe + latent_dim paramétrable + sauvegarde métriques

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from tqdm import tqdm
import os, json, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------
# Helpers
# -------------------------
def to_python(obj):
    """Recursively convert numpy / torch types to native Python for JSON serialization."""
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

def dataset_exists(dataset_name, data_dir):
    data_dir = Path(data_dir)
    if dataset_name.upper() == "MNIST":
        required = ["MNIST/raw/train-images-idx3-ubyte",
                    "MNIST/raw/train-labels-idx1-ubyte"]
    else:
        required = ["KMNIST/raw/train-images-idx3-ubyte",
                    "KMNIST/raw/train-labels-idx1-ubyte"]
    return all((data_dir / r).exists() for r in required)

class TwoCropsTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)

def balanced_subset_indices(dataset, per_class, seed=SEED):
    labels = np.array([y for _, y in dataset])
    selected = []
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        selected.extend(idx[:per_class])
    return selected

# -------------------------
# Model components (IIC-style)
# -------------------------
class IICBackbone(nn.Module):
    """Small CNN backbone appropriate for 28x28 greyscale images."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,stride=2,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,stride=2,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.out_h, self.out_w, self.out_c = 7,7,128
        self.out_dim = self.out_c * self.out_h * self.out_w

    def forward(self, x):
        f = self.features(x)
        return f.view(f.size(0), -1)

class IICHead(nn.Module):
    def __init__(self, dim_in, num_clusters, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, num_clusters)
        )
    def forward(self, z):
        return F.softmax(self.net(z), dim=1)

class IICModelPaper(nn.Module):
    """IIC model with multi-heads and multi-resolution outputs."""
    def __init__(self, num_clusters=10, over_factor=10, latent_dim=512):
        super().__init__()
        self.backbone = IICBackbone()
        dim_full = self.backbone.out_dim
        self.head_main = IICHead(dim_full, num_clusters, latent_dim=latent_dim)
        self.head_over = IICHead(dim_full, num_clusters*over_factor, latent_dim=latent_dim)
        self.pool_small = nn.AdaptiveAvgPool2d((3,3))
        dim_small = 128*3*3
        self.head_main_small = IICHead(dim_small, num_clusters, latent_dim=latent_dim)
        self.head_over_small = IICHead(dim_small, num_clusters*over_factor, latent_dim=latent_dim)

    def forward(self, x):
        f = self.backbone.features(x)
        z_full = f.view(f.size(0), -1)
        p_main = self.head_main(z_full)
        p_over = self.head_over(z_full)
        f_small = self.pool_small(f)
        z_small = f_small.view(f_small.size(0), -1)
        p_main_small = self.head_main_small(z_small)
        p_over_small = self.head_over_small(z_small)
        return {
            "main": p_main,
            "over": p_over,
            "main_small": p_main_small,
            "over_small": p_over_small,
            "z_full": z_full,
            "z_small": z_small
        }

# -------------------------
# Loss
# -------------------------
def mutual_info_term(p1, p2, eps=1e-10):
    P = torch.matmul(p1.t(), p2)
    P = P / (P.sum() + eps)
    pi = P.sum(dim=1).view(-1,1)
    pj = P.sum(dim=0).view(1,-1)
    mi = (P * (torch.log(P+eps) - torch.log(pi+eps) - torch.log(pj+eps))).sum()
    return -mi

def iic_total_loss(out1, out2, lambda_entropy=2.4):
    keys = ["main","over","main_small","over_small"]
    losses = []
    for k in keys:
        p1, p2 = out1[k], out2[k]
        mi_neg = mutual_info_term(p1,p2)
        marginal = 0.5*(p1.mean(dim=0)+p2.mean(dim=0))
        entropy = (marginal*torch.log(marginal+1e-10)).sum()
        losses.append(mi_neg + lambda_entropy*entropy)
    return torch.stack(losses).mean()

# -------------------------
# Training
# -------------------------
def train_epoch(model, loader, opt, device, lambda_entropy):
    model.train()
    total_loss, n = 0.0, 0
    for (xb, _) in loader:
        x1,x2 = xb
        x1,x2 = x1.to(device), x2.to(device)
        out1, out2 = model(x1), model(x2)
        loss = iic_total_loss(out1, out2, lambda_entropy)
        opt.zero_grad(); loss.backward(); opt.step()
        batch_size = x1.size(0)
        total_loss += loss.item()*batch_size
        n += batch_size
    return total_loss/max(1,n)

# -------------------------
# Extract embeddings & predictions
# -------------------------
def extract_embeddings_and_preds(model, dataset, device='cpu'):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    feats, labels, preds = [], [], []
    with torch.no_grad():
        for xb,y in loader:
            x = xb[0] if isinstance(xb,(list,tuple)) else xb
            x = x.to(device)
            out = model(x)
            pred = out["main"].argmax(dim=1).cpu().numpy()
            feats.append(out["z_full"].cpu().numpy())
            labels.append(y.numpy())
            preds.append(pred)
    feats = np.concatenate(feats,axis=0) if feats else np.empty((0,model.backbone.out_dim))
    labels = np.concatenate(labels,axis=0) if labels else np.empty((0,),dtype=np.int64)
    preds = np.concatenate(preds,axis=0) if preds else np.empty((0,),dtype=np.int64)
    return feats, labels, preds

# -------------------------
# Metrics
# -------------------------
def compute_metrics_from_preds(feats, true_labels, pred_labels):
    d = {}
    try: d['ARI'] = float(ARI(true_labels,pred_labels)); d['NMI']=float(NMI(true_labels,pred_labels))
    except: d['ARI'],d['NMI']=float('nan'),float('nan')
    try: d['Silhouette']=float(silhouette_score(feats,pred_labels))
    except: d['Silhouette']=float('nan')
    try: d['Calinski_Harabasz']=float(calinski_harabasz_score(feats,pred_labels))
    except: d['Calinski_Harabasz']=float('nan')
    try: d['Davies_Bouldin']=float(davies_bouldin_score(feats,pred_labels))
    except: d['Davies_Bouldin']=float('nan')
    return d

# -------------------------
# t-SNE
# -------------------------
def plot_tsne(feats, labels, title="t-SNE"):
    if len(feats)==0: return
    tsne = TSNE(n_components=2, random_state=SEED)
    emb = tsne.fit_transform(feats)
    classes = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(7,7))
    for i, cls in enumerate(classes):
        idx = labels==cls
        plt.scatter(emb[idx,0], emb[idx,1], c=[cmap(i%10)], s=6, label=str(cls), alpha=0.8)
    plt.title(title)
    plt.legend(title="Classes", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()

# -------------------------
# Main
# -------------------------
def parse_args():
    script_dir = Path(__file__).parent.resolve()  # dossier du script
    default_data_dir = script_dir / "data"
    
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST","KMNIST"])
    #p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--data_dir", type=str, default=str(default_data_dir))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_entropy", type=float, default=2.4)
    p.add_argument("--over_factor", type=int, default=10)
    p.add_argument("--latent_dim", type=int, default=512, help="Latent space dimension")
    p.add_argument("--per_class", type=int, default=1000)
    p.add_argument("--n_samples", type=int, default=60000)
    p.add_argument("--save_dir", type=str, default="./iic_runs")
    p.add_argument("--tsne", type=bool, default=True)
    return p.parse_args()

def mainiccminst():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    run_name = datetime.now().strftime("IIC_paper_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print("Device:",device)

    normalize = transforms.Normalize([0.5], [0.5])
    augment = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5
    ),
    transforms.ToTensor(),
    normalize,
    transforms.RandomErasing(p=0.1)])
    two = TwoCropsTransform(augment)

    # Load dataset
    data_exists = dataset_exists(args.dataset,args.data_dir)
    if args.dataset.upper()=="MNIST":
        base = datasets.MNIST(args.data_dir,train=True,download=not data_exists,transform=two)
        num_clusters=10
    else:
        base = datasets.KMNIST(args.data_dir,train=True,download=not data_exists,transform=two)
        num_clusters=10
    print("Dataset samples:",len(base))

    # optional subsample
    if args.n_samples and args.n_samples<len(base):
        idx=np.arange(len(base)); np.random.shuffle(idx)
        base=Subset(base,idx[:args.n_samples])
        print("Subsampled to",args.n_samples)

    # balanced subset
    dataset = Subset(base,balanced_subset_indices(base,args.per_class)) if args.per_class else base
    labels_check = np.array([y for _,y in dataset])
    for u,c in zip(*np.unique(labels_check,return_counts=True)):
        print(f"class {u}: {c}")

    # Model
    model = IICModelPaper(num_clusters=num_clusters,over_factor=args.over_factor,latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=False)

    # Training
    print("Start training IIC...")
    for ep in range(1,args.epochs+1):
        loss_epoch = train_epoch(model,loader,opt,device,args.lambda_entropy)
        if ep%10==0 or ep==1 or ep==args.epochs:
            print(f"Epoch {ep}/{args.epochs} — loss: {loss_epoch:.5f}")

        if ep%50==0 or ep==args.epochs:
            feats, labels, preds = extract_embeddings_and_preds(model,dataset,device=device)
            km_preds = KMeans(n_clusters=num_clusters,random_state=SEED).fit(feats).labels_
            metrics_model = compute_metrics_from_preds(feats,labels,preds)
            metrics_km = compute_metrics_from_preds(feats,labels,km_preds)
            print(f"Eval @ epoch {ep}: ARI(model)={metrics_model['ARI']:.4f}, NMI(model)={metrics_model['NMI']:.4f}")
            # Save metrics JSON
            out={"epoch":int(ep),"metrics_model":metrics_model,"metrics_km":metrics_km}
            with open(os.path.join(out_dir,f"metrics_ep{ep}.json"),"w") as f:
                json.dump(to_python(out),f,indent=2)

    # Final evaluation
    feats, labels, preds = extract_embeddings_and_preds(model,dataset,device=device)
    km_preds = KMeans(n_clusters=num_clusters,random_state=SEED).fit(feats).labels_
    metrics_model = compute_metrics_from_preds(feats,labels,preds)
    metrics_km = compute_metrics_from_preds(feats,labels,km_preds)
    print("\n===== FINAL METRICS (IIC) =====")
    for k,v in metrics_model.items(): print(f"{k}: {v:.4f}")
    print("\n===== FINAL METRICS (KMeans) =====")
    for k,v in metrics_km.items(): print(f"{k}: {v:.4f}")
    out_final={"metrics_model":metrics_model,"metrics_km":metrics_km,"label_distribution":{int(u):int(c) for u,c in zip(*np.unique(labels,return_counts=True))}}
    with open(os.path.join(out_dir,"metrics_final.json"),"w") as f:
        json.dump(to_python(out_final),f,indent=2)

    # t-SNE
    if args.tsne:
        print("Plotting t-SNE...")
        plot_tsne(feats,labels,title=f"t-SNE {args.dataset} (true)")
        plot_tsne(feats,preds,title=f"t-SNE {args.dataset} (IIC preds)")

    print("Done. Outputs in:",out_dir)

def launchiccmnist():
    mainiccminst()
