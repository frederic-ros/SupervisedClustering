# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:15:20 2025

@author: frederic.ros
"""
from argparse import ArgumentParser, Namespace
from F_formes import generate_clusters,generate_nonlinear_clusters,generate_mixed_clusters_separable_1, generate_clusters_with_different_densities,generate_heterogeneous_clusters
from F_genereOld import CreateData

def get_config_synthetic() -> Namespace:
    parser = ArgumentParser()

   
    parser.add_argument("--op0_proba_St", type=float, default=20)
    parser.add_argument("--op0_proba_An", type=float, default=50)
    parser.add_argument("--op0_Proba_all", type=float, default=80)
    parser.add_argument("--op0_Score_min", type=float, default=0.9)
    
    parser.add_argument("--op1_min_blob_size", type=int, default=50)
    parser.add_argument("--op1_p_noise", type=float, default=0.05)
    
    parser.add_argument("--op2_delta_density", type=float, default=5.0)
    
    parser.add_argument("--op3_p_noise", type=float, default=0.05)
    parser.add_argument("--op3_delta_density", type=float, default=2)
    parser.add_argument("--op3_min_cluster_size", type=int, default=30)
    
    parser.add_argument("--op4_separability", type=float, default=2.0)
    
    parser.add_argument("--op5_separability", type=float, default=3.0)
    parser.add_argument("--op5_nonlinearity", type=float, default=0.5)
    
    
    return parser.parse_args()

def SyntheticSet(option, n_samples=1000, dim = 2, max_dev=0.3, p_noise=0.05, 
                 hamming_distance=0.1,
                 n_clusters=4, index_random=0):
    
    cfg = get_config_synthetic()
    
    
    
    if option == 0: X, y, x_pur,ypur,valid,S = CreateData(n_samples=n_samples, dim=dim, hamming_distance=hamming_distance,
                                                          p_noise=p_noise, noise_t=1,
                                              max_dev=max_dev,n_centers=n_clusters, 
                                              Score_min=cfg.op0_Score_min, 
                                              proba_St=cfg.op0_proba_St, 
                                              proba_An=cfg.op0_proba_An,
                                              Proba_all = cfg.op0_Proba_all, Draw=0)
        
    if option == 1: X, y, y = generate_mixed_clusters_separable_1(n_items=n_samples, proj_dim=dim, min_blob_size=cfg.op1_min_blob_size, 
                                                                  p_noise=cfg.op1_p_noise, random_state=0+index_random)
        
    if option == 2: X,y = generate_clusters_with_different_densities(n_samples=n_samples, p_noise=p_noise, n_clusters=n_clusters, 
                                                                         delta_density=cfg.op2_delta_density, proj_dim=dim, random_state=42+index_random)
    
    if option == 3: X,y = generate_heterogeneous_clusters(n_samples=n_samples, p_noise=cfg.op3_p_noise, n_clusters=n_clusters, 
                                                          delta_density=cfg.op3_delta_density,min_cluster_size=cfg.op3_min_cluster_size, 
                                                          proj_dim=dim, random_state=0+index_random)
    
    if option == 4: X,y = generate_clusters(n_samples=n_samples, n_clusters=n_clusters, 
                                            dim=dim, separability=cfg.op4_separability, 
                                            random_state=index_random)
    
    if option == 5: X,y = generate_nonlinear_clusters(n_samples=n_samples, n_clusters=n_clusters, 
                                    dim=dim, separability=cfg.op5_separability, nonlinearity=cfg.op5_nonlinearity, random_state=42)

    return X,y
