# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:48:49 2025

@author: frederic.ros
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class ProximityPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128,64], 
                 dropout=0.2, lr=1e-3, device=None):
        super(ProximityPredictor, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        # MLP simple
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())  # sortie entre 0 et 1

        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    import matplotlib.pyplot as plt
    import matplotlib

    def fit(self, X_train, y_train, X_val=None, y_val=None, 
        epochs=50, batch_size=32, verbose=True, plot_batch=False):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None

      
        # --- Initialisation du graphique dynamique ---
        all_batch_losses = []
        if plot_batch:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8,5))
            scatter = ax.scatter([], [], c='blue', s=10)
            line, = ax.plot([], [], 'r-', linewidth=2)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Curve')
            ax.grid(True)

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                all_batch_losses.append(loss.item())

                if plot_batch and epoch == 0: #to display...
                    # Mise à jour des données sans effacer les artistes
                    scatter.set_offsets(
                        np.c_[range(len(all_batch_losses)), all_batch_losses]
                        )

                    if len(all_batch_losses) >= len(train_loader):
                        last_epoch_losses = all_batch_losses[-len(train_loader):]
                        line.set_data(
                            range(len(all_batch_losses)-len(train_loader), len(all_batch_losses)),
                        last_epoch_losses
                        )

                    ax.relim()
                    ax.autoscale_view()
                    ax.set_title(f'Learning Curve - Epoch {epoch+1}')
                    plt.pause(0.05)


            # Log par epoch
            if verbose:
                avg_epoch_loss = sum(all_batch_losses[-len(train_loader):]) / len(train_loader)
                log = f"Epoch {epoch+1}/{epochs} - Avg Train Loss: {avg_epoch_loss:.4f}"
                if val_loader:
                    val_loss = self.evaluate_loader(val_loader)
                    log += f" - Val Loss: {val_loss:.4f}"
                print(log)

        if plot_batch:
            plt.ioff()
            plt.show()
    
    def evaluate_loader(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for xb, yb in data_loader:
                preds = self.model(xb)
                total_loss += self.criterion(preds, yb).item()
        return total_loss / len(data_loader)

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X)
        return preds.cpu().numpy()
  

    def evaluate(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = DataLoader(TensorDataset(X, y), batch_size=32)
        return self.evaluate_loader(dataset)

    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)    

def launchmodel(in_pat, out_pat, epochs=200):
    input_dim = in_pat.shape[1]
    output_dim = out_pat.shape[1]
    
    val_inputs = train_inputs = in_pat
    val_outputs = train_outputs = out_pat
    
    model = ProximityPredictor(input_dim, output_dim)

    # Training (avec ou sans validation)
    model.fit(train_inputs, train_outputs, X_val=val_inputs, y_val=val_outputs, epochs=epochs)

    # Évaluation
    loss = model.evaluate(val_inputs, val_outputs)
    print("Validation Loss:", loss)

    # Prédictions
    '''
    predictions = model.predict(train_inputs)
    np.savetxt("prediction.txt", predictions, delimiter="\t", fmt="%.6f")
    '''
    # Sauvegarde / chargement
    
    return model
    