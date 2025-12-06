# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:08:30 2025

@author: frederic.ros
"""
import numpy as np
from F_newidee import genpoint, compute_onlyinput_patterns_with_embedding
from F_modelenewidee import launchmodel, ProximityPredictor
from F_mutualScan import cluster_mutual_density, visualize_clusters_2d, cluster_mutual_density_from_density
from F_processreal import Load_reeldata, getPCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def plot_density_with_projection(X, densities, method="tsne", title="Projection avec densité", cmap="viridis", random_state=42):
    """
    Projette les données X en 2D avec t-SNE ou UMAP, et affiche les points colorés selon leur densité.

    Parameters:
    - X : ndarray (N, D), données d'entrée
    - densities : ndarray (N,), densité associée à chaque point (valeurs entre 0 et 1)
    - method : str, "tsne" ou "umap"
    - title : str, titre du graphique
    - cmap : str, colormap matplotlib
    - random_state : int, pour la reproductibilité
    """
    if method == "tsne":
        projector = TSNE(n_components=2, random_state=random_state, perplexity=30)
    if method == "umap":
        projector = umap.UMAP(n_components=2, random_state=random_state)
    
    X_2d = projector.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=densities, cmap=cmap, s=30, edgecolor='k')
    plt.colorbar(scatter, label='Densité')
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel('Projection 1')
    plt.ylabel('Projection 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def TestModelReal(directory_path = "realdata",p_embedding=10,namemodel="model"):
    Kv_in = 16
    Kv_out = 16
    draw = False
    
    
    input_dim = Kv_in * Kv_in + p_embedding 
    output_dim = Kv_out
    model = ProximityPredictor(input_dim=input_dim, output_dim=output_dim)
    model.load_model(namemodel)
    
    files, data_l, y_l = Load_reeldata(directory_path = directory_path,t = 10, start = 0)
    for i in range(0,len(y_l)):
        print(files[i])
        if draw== True: visualize_clusters_2d(data_l[i], y_l[i],method='tsne' )
        if data_l[i].shape[1] <= 10: 
            X = data_l[i]
        else : 
            X = getPCA(data_l[i])
            
        in_pat = compute_onlyinput_patterns_with_embedding(X, p_embedding=p_embedding, Kv_in=Kv_in, 
                                                               embedding_dict=None, random_state=None,
                                                               normalize_distances=False)
        visualize_clusters_2d(X, y_l[i],method='tsne',name=files[i] + "true labels (dim="+str(data_l[i].shape[1])+")" )       
        memberhsip = model.predict(in_pat)
        labels, densities, G = cluster_mutual_density(X, memberhsip, Kv=16, threshold=0.5, min_density=0.7, normalize=True)
        if draw == True: plot_density_with_projection(X, densities, method="tsne", 
                                                          title="Projection avec densité", cmap='Reds', random_state=42)
        visualize_clusters_2d(X, labels,method='tsne',name=files[i] + "(dim="+str(data_l[i].shape[1])+")" )


#######################################################################################   
def TestModelSynthetique(n=1,dim=2, p_embedding=10,option=0,namemodel="model"):
    Kv_in = 16
    Kv_out = 16
    draw = False
    Competiteurs = True
    Our = False
    
    run = np.zeros(15,int)
    run[13] = 1
    '''
    run = np.ones(15,int)
    '''
    noise_methode = np.zeros(15,int)
    noise = 0
    
    input_dim = Kv_in * Kv_in + p_embedding 
    output_dim = Kv_out
    model = ProximityPredictor(input_dim=input_dim, output_dim=output_dim)
    model.load_model(namemodel)
    for i in range(n):
        X,y = SyntheticSet(option=option, n_samples=1000, dim = 2, p_noise=0.05, n_clusters=4, index_random=i)
        in_pat = compute_onlyinput_patterns_with_embedding(X, p_embedding=p_embedding, Kv_in=Kv_in, 
                                                               embedding_dict=None, random_state=None,
                                                               normalize_distances=False)
               
        memberhsip = model.predict(in_pat)
        labels, densities, G = cluster_mutual_density(X, memberhsip, Kv=16, threshold=0.5, min_density=0.6, normalize=True)
        if draw == True: plot_density_with_projection(X, densities, method="tsne", 
                                     title="Projection avec densité", cmap='Reds', random_state=42)
        visualize_clusters_2d(X, labels,method='tsne' )
        