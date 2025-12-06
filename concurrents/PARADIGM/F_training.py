# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:05:30 2025

@author: frederic.ros
"""
import numpy as np
from F_newidee import genpoint
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from F_modelenewidee import launchmodel, ProximityPredictor
from F_mutualScan import cluster_mutual_density, visualize_clusters_2d, cluster_mutual_density_from_density
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import joblib

def save_train_data_output(filename="data.txt", train_output=None):
    """
    Sauvegarde train_input et train_output dans un fichier texte tabulé.
    Chaque ligne contient : [features... | outputs...]
    """
    
    # Concaténation des colonnes input + output
    data = train_output
    
    # Sauvegarde avec tabulations
    np.savetxt(filename, data, delimiter='\t', fmt='%.6f')
    print(f"✅ Données sauvegardées dans '{filename}' ({data.shape[0]} lignes, {data.shape[1]} colonnes)")

def save_train_data(filename="data.txt", train_input=None, train_output=None):
    """
    Sauvegarde train_input et train_output dans un fichier texte tabulé.
    Chaque ligne contient : [features... | outputs...]
    """
    if train_input.shape[0] != train_output.shape[0]:
        raise ValueError("Les deux matrices doivent avoir le même nombre d'échantillons (lignes).")
    
    # Concaténation des colonnes input + output
    data = np.hstack((train_input, train_output))
    
    # Sauvegarde avec tabulations
    np.savetxt(filename, data, delimiter='\t', fmt='%.6f')
    print(f"✅ Données sauvegardées dans '{filename}' ({data.shape[0]} lignes, {data.shape[1]} colonnes)")

def load_train_data(filename="data.txt", n_inputs=None, n_outputs=None):
    """
    Charge un fichier tabulé (sauvé par save_train_data) et sépare
    les entrées (train_input) et les sorties (train_output).
    
    ⚙️ Paramètres :
      - filename : chemin du fichier .txt
      - n_inputs : nombre de colonnes correspondant aux inputs
      - n_outputs : nombre de colonnes correspondant aux outputs
      
    ➕ Il faut spécifier **au moins un** des deux (n_inputs ou n_outputs).
    """
    data = np.loadtxt(filename, delimiter='\t')
    n_cols = data.shape[1]

    # Vérification de la cohérence
    if n_inputs is None and n_outputs is None:
        raise ValueError("Tu dois préciser soit n_inputs, soit n_outputs.")
    
    if n_inputs is not None:
        n_outputs = n_cols - n_inputs
    elif n_outputs is not None:
        n_inputs = n_cols - n_outputs

    if n_inputs + n_outputs != n_cols:
        raise ValueError("Dimensions incohérentes entre inputs/outputs et fichier.")

    # Séparation des matrices
    train_input = data[:, :n_inputs]
    train_output = data[:, n_inputs:]
    
    print(f"✅ Données chargées depuis '{filename}' : "
          f"{data.shape[0]} lignes, {n_inputs} inputs, {n_outputs} outputs.")
    
    return train_input, train_output


def add_to_training_set(train_inputs, train_outputs, new_inputs, new_outputs):
    """
    Ajoute de nouveaux patterns à la base d'apprentissage existante.

    Args:
    - train_inputs (ndarray or None): Inputs existants (ou None pour initialisation).
    - train_outputs (ndarray or None): Outputs existants (ou None pour initialisation).
    - new_inputs (ndarray): Nouveaux inputs à ajouter.
    - new_outputs (ndarray): Nouveaux outputs à ajouter.

    Returns:
    - updated_inputs (ndarray): Inputs concaténés.
    - updated_outputs (ndarray): Outputs concaténés.
    """
    if train_inputs is None or train_outputs is None:
        return new_inputs, new_outputs
    else:
        updated_inputs = np.concatenate([train_inputs, new_inputs], axis=0)
        updated_outputs = np.concatenate([train_outputs, new_outputs], axis=0)
        return updated_inputs, updated_outputs

def save_model(model, filename):
    """Sauvegarde un modèle entraîné dans un fichier .pkl"""
    joblib.dump(model, filename)
    print(f"✅ Modèle sauvegardé dans '{filename}'")

def load_model(filename):
    """Recharge un modèle sauvegardé"""
    model = joblib.load(filename)
    print(f"✅ Modèle chargé depuis '{filename}'")
    return model

def CreateModel(n=1000, n_samples = 1000,dim=2, dev_training=0.3,hamming_training=0.01,
                noise_training=0.1,
                p_embedding=10, training_data="data.txt",namemodel="model",
                epoch=200,option=0,use_fuzzy=True):
    Kv_in = 16
    Kv_out = 16
    train_inputs=None 
    train_outputs=None
    print("data base creation")
    for i in range(n): #number if each space p
        for j in range(len(dim)): #for each dimension.
            X,in_pat, out_pat = genpoint(n_samples=n_samples, dim=dim[j], max_dev=dev_training,
                                         hamming_distance=hamming_training,
                                         p_noise=noise_training,p_embedding = p_embedding,
                                         Kv_in = Kv_in, Kv_out = Kv_out, K_m = 5,Draw = False, 
                                         filtered = True, save = False,use_fuzzy = use_fuzzy)
            train_inputs, train_outputs = add_to_training_set(train_inputs, train_outputs,in_pat, out_pat)

    save_train_data(filename=training_data, train_input=train_inputs, 
                    train_output=train_outputs)
    save_train_data_output(filename="out"+training_data, train_output=train_outputs)
    


    print("train the model")
    if option==0:
        model = launchmodel(train_inputs, train_outputs,epochs=epoch)
        model.save_model(namemodel+".pth")
    else:
        model = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=300,
                                                   learning_rate=0.05,
                                                   num_leaves=31,
                                                   n_jobs=-1    ))
        model.fit(train_inputs, train_outputs)
        y_pred = model.predict(train_inputs)
        r2 = r2_score(train_outputs, y_pred)
        rmse = np.sqrt(mean_squared_error(train_outputs, y_pred))        
        mae = mean_absolute_error(train_outputs, y_pred)
        print(f"R² : {r2:.3f} | RMSE : {rmse:.3f} | MAE : {mae:.3f}")
        save_model(model, namemodel + "r"+ "pkl")
        
        model = RandomForestRegressor(n_estimators=100,max_depth=None,
                                  n_jobs=-1,random_state=42)
        model.fit(train_inputs, train_outputs)
        y_pred = model.predict(train_inputs)
        r2 = r2_score(train_outputs, y_pred)
        rmse = np.sqrt(mean_squared_error(train_outputs, y_pred))        
        mae = mean_absolute_error(train_outputs, y_pred)
        print(f"R² : {r2:.3f} | RMSE : {rmse:.3f} | MAE : {mae:.3f}")
        save_model(model, namemodel + "f"+ "pkl")
        