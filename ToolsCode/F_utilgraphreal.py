# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:58:50 2025

@author: frederic.ros
"""
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

import os
import glob
import numpy as np

import os
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

def plot_stacked_bars(p1, p2, p3, labels=None):
    """
    Displays a stacked bar chart showing the win percentages
    for each method across the three evaluation metrics (NMI, ARI, CVI).

    Parameters:
        p1, p2, p3 : dict
            Dictionaries {method_id: percentage} for each metric.
        labels : list of str or None, optional
            Custom labels for methods. Can contain None values.
            Only non-None labels appear in the legend.
            Length must match the number of methods.
    """

    # Combine data
    df = pd.DataFrame({
        'NMI': p1,
        'ARI': p2,
        'CVI': p3
    }).fillna(0).sort_index()

    df_T = df.T

    # Check labels
    if labels is not None:
        if len(labels) != len(df_T.columns):
            raise ValueError(f"Length of labels ({len(labels)}) does not match number of methods ({len(df_T.columns)})")
        label_map = {col: labels[i] for i, col in enumerate(df_T.columns)}
    else:
        label_map = {col: f"Method {col}" for col in df_T.columns}

    # Define a distinct color palette
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(df_T.columns))]

    plt.figure(figsize=(10, 6))
    bottoms = pd.Series([0.0] * len(df_T), index=df_T.index)

    for idx, method_id in enumerate(df_T.columns):
        values = df_T[method_id]
        label = label_map.get(method_id)
        # If label is None, it won't appear in the legend
        plt.bar(
            df_T.index,
            values,
            bottom=bottoms,
            color=colors[idx],
            label=label if label is not None else "_nolegend_"
        )
        bottoms += values

    # Add titles and axes
    plt.title("Win Percentage Distribution per Metric", fontsize=14, weight='bold')
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Cumulative Win Percentage (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Show totals on top of bars
    for i, metric in enumerate(df_T.index):
        total = df_T.loc[metric].sum()
        plt.text(i, total + 1, f"{total:.1f}%", ha='center', va='bottom', color='black', fontsize=10)

    # Only display legend if there are visible labels
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    if any(label != "_nolegend_" for label in legend_labels):
        plt.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def mean_std_by_method(folder_path, prefix, file_nums=[0,1],
                       column_indices=[0], line_nums=None):
    """
    Calcule la moyenne et l'écart-type par méthode (fichier) et par colonne,
    sur les lignes sélectionnées.

    Args:
        folder_path (str): Dossier contenant les fichiers
        prefix (str): Préfixe commun aux fichiers
        file_nums (list[int]): Numéros des fichiers/méthodes à considérer
        column_indices (list[int]): Indices des colonnes à traiter
        line_nums (list[int] or None): Indices des lignes à inclure (datasets).
                                       Si None, toutes les lignes sont utilisées.

    Returns:
        tuple:
            means : np.ndarray de forme (n_files, n_columns)
            stds  : np.ndarray de forme (n_files, n_columns)
    """
    all_means = []
    all_stds = []

    for num in file_nums:
        file_pattern = os.path.join(folder_path, f"{prefix}{num}*.txt")
        file_list = sorted(glob.glob(file_pattern))

        if len(file_list) == 0:
            print(f"Aucun fichier trouvé pour {prefix}{num}")
            continue

        file_path = file_list[0]
        data = np.loadtxt(file_path)

        # Sélection des lignes si précisé
        if line_nums is not None:
            valid_lines = [i for i in line_nums if i < data.shape[0]]
            data = data[valid_lines, :]

        # Calculs colonne par colonne
        file_means = []
        file_stds = []

        for col in column_indices:
            if col < data.shape[1]:
                col_values = data[:, col]
                file_means.append(np.mean(col_values))
                file_stds.append(np.std(col_values))
            else:
                file_means.append(np.nan)
                file_stds.append(np.nan)

        all_means.append(file_means)
        all_stds.append(file_stds)

    return np.array(all_means), np.array(all_stds)


def count_max_per_file_percent(folder_path, prefix, file_nums=[0,1], column_index=0, line_nums=None):
    """
    For a given column, count how many times each file has the maximum value across
    the selected files, restricted to selected line indices (line_nums).

    Args:
        folder_path (str): Path to folder containing files.
        prefix (str): Common prefix for files (e.g. 'prefix0.txt', 'prefix1.txt').
        file_nums (list of int): List of file numbers to compare (e.g. [0,1,2]).
        column_index (int): Index of the column to analyze.
        line_nums (list of int or None): Line indices to compare.
                                         If None, all available lines are used.

    Returns:
        valid_files (list): File numbers actually used.
        counts (dict): {file_num: number of wins}.
        percents (dict): {file_num: % of wins}.
        n_lines_considered (int): Number of lines considered.
    """
    data_list = []
    valid_files = []

    # Charger les fichiers correspondants
    for num in file_nums:
        file_pattern = os.path.join(folder_path, f"{prefix}{num}*.txt")
        file_list = sorted(glob.glob(file_pattern))
        if len(file_list) == 0:
            print(f"No file found for {prefix}{num}")
            continue

        data = np.loadtxt(file_list[0])
        if column_index >= data.shape[1]:
            print(f"Column {column_index} not found in {prefix}{num}")
            continue

        data_list.append(data[:, column_index])
        valid_files.append(num)

    if len(data_list) == 0:
        raise ValueError("No valid files found.")

    # Longueur minimale commune
    min_len = min(len(arr) for arr in data_list)

    # Sélectionner les lignes demandées
    if line_nums is None:
        indices = np.arange(min_len)
    else:
        indices = [i for i in line_nums if 0 <= i < min_len]
        if len(indices) == 0:
            raise ValueError("No valid line indices within data range.")

    # Comparer uniquement les lignes sélectionnées
    all_data = np.vstack([arr[indices] for arr in data_list])  # (n_files, n_selected_lines)

    n_files = len(valid_files)
    counts = np.zeros(n_files)

    # Comparer chaque ligne
    for j in range(all_data.shape[1]):
        max_val = np.max(all_data[:, j])
        winners = np.where(all_data[:, j] == max_val)[0]
        share = 1.0 / len(winners)
        counts[winners] += share

    # Conversion en pourcentages
    n_lines_considered = len(indices)
    percents = (counts / n_lines_considered) * 100.0

    # Dictionnaires de sortie
    counts_dict = {valid_files[i]: counts[i] for i in range(n_files)}
    percents_dict = {valid_files[i]: percents[i] for i in range(n_files)}

    return valid_files, counts_dict, percents_dict, n_lines_considered

'''
def plot_column_two_files(folder_path, prefix, file_nums=[0,1], column_index=0,
                          title="", ylabel="Value", legend_fontsize=10,
                          line_nums=None):
    """
    Plot a specific column from two selected files with the same prefix.
    Markers are shown only for the line indices specified in 'line_nums'.
    
    Args:
        folder_path (str): Path to folder containing files
        prefix (str): Filename prefix to select files
        file_nums (list of int): List of two file numbers to plot (e.g., [0,1])
        column_index (int): Index of the column to plot
        title (str): Plot title
        ylabel (str): Y-axis label
        legend_fontsize (int or float): Font size of the legend
        line_nums (list of int or None): Indices of lines to display markers. If None, show no markers.
    """
    if len(file_nums) != 2:
        raise ValueError("file_nums must contain exactly 2 file numbers.")

    plt.figure(figsize=(8,5))

    for num in file_nums:
        # Chercher le fichier correspondant au numéro
        file_pattern = os.path.join(folder_path, f"{prefix}{num}*.txt")
        file_list = sorted(glob.glob(file_pattern))
        
        if len(file_list) == 0:
            print(f"No file found for {prefix}{num}")
            continue

        file_path = file_list[0]  # on prend le premier correspondant
        data = np.loadtxt(file_path)
        values = data[:, column_index]  # colonne choisie

        x = np.arange(len(values))

        # Tracer la courbe complète
        plt.plot(x, values, linestyle='-', label=f"{prefix}{num}")

        # Tracer les markers seulement pour les lignes sélectionnées
        if line_nums is not None:
            selected_x = [i for i in line_nums if i < len(values)]
            selected_y = [values[i] for i in selected_x]
            plt.scatter(selected_x, selected_y, marker='o', color=plt.gca().lines[-1].get_color())

    plt.xlabel("Line index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()
'''
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_column_k_files(folder_path, prefix, file_nums=[0,1], column_index=0,
                        title="", ylabel="Value", legend_fontsize=10,
                        line_nums=None, labels=None):
    """
    Plot only the selected points from a specific column in multiple files 
    and connect them with lines.

    Args:
        folder_path (str): Path to folder containing files
        prefix (str): Filename prefix to select files
        file_nums (list of int): List of file numbers to plot (e.g., [0,1,2])
        column_index (int): Index of the column to plot
        title (str): Plot title
        ylabel (str): Y-axis label
        legend_fontsize (int or float): Font size of the legend
        line_nums (list of int or None): Indices of lines to display and connect.
                                         If None, connect all lines.
        labels (list of str or None): Custom labels for each file. 
                                      Must have the same length as file_nums.
                                      Use None to skip legend entry for that file.
    """
    if not isinstance(file_nums, (list, tuple)) or len(file_nums) == 0:
        raise ValueError("file_nums must be a non-empty list or tuple of integers.")
    
    if labels is not None and len(labels) != len(file_nums):
        raise ValueError("If provided, 'labels' must have the same length as 'file_nums'.")

    plt.figure(figsize=(8,5))

    for idx, num in enumerate(file_nums):
        # Trouver le fichier correspondant
        file_pattern = os.path.join(folder_path, f"{prefix}{num}*.txt")
        file_list = sorted(glob.glob(file_pattern))
        
        if len(file_list) == 0:
            print(f"No file found for {prefix}{num}")
            continue

        file_path = file_list[0]
        data = np.loadtxt(file_path)
        values = data[:, column_index]

        # Sélection des indices
        if line_nums is not None:
            selected_x = [i for i in line_nums if i < len(values)]
            selected_y = [values[i] for i in selected_x]
        else:
            selected_x = np.arange(len(values))
            selected_y = values

        # Déterminer le label
        label = None
        if labels is not None:
            label = labels[idx]
        else:
            label = f"{prefix}{num}"

        # Tracer — ignorer la légende si label=None
        plt.plot(selected_x, selected_y, marker='o', linestyle='-',
                 label=label if label is not None else "_nolegend_")

    plt.xlabel("Data Set")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    
    # Afficher la légende uniquement si au moins un label visible
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    if any(l != "_nolegend_" for l in legend_labels):
        plt.legend(fontsize=legend_fontsize)
    
    plt.tight_layout()
    plt.show()



def plot_two_features(folder_path, prefix, feature1=3, feature2=4, title="", ylabel="Value"):
    """
    Plot two features from the first line of multiple text files with a given prefix.
    
    Args:
        folder_path (str): Path to folder containing files
        prefix (str): Filename prefix to select files
        feature1 (int): Index of the first feature to display
        feature2 (int): Index of the second feature to display
        title (str): Plot title
        ylabel (str): Y-axis label
    """
    file_pattern = os.path.join(folder_path, f"{prefix}*.txt")
    file_list = sorted(glob.glob(file_pattern))
    
    if len(file_list) == 0:
        raise ValueError(f"No files found starting with '{prefix}' in {folder_path}.")

    x_vals = []
    f1_vals = []
    f2_vals = []

    for file in file_list:
        basename = os.path.basename(file)
        num_str = ''.join(filter(str.isdigit, basename))
        if not num_str:
            continue
        num = int(num_str)
        x_vals.append(num)

        data = np.loadtxt(file)
        f1_vals.append(data[0, feature1])
        f2_vals.append(data[0, feature2])

    # Trier selon les numéros de fichiers
    x_vals, f1_vals, f2_vals = zip(*sorted(zip(x_vals, f1_vals, f2_vals)))

    plt.figure(figsize=(8,5))
    plt.plot(x_vals, f1_vals, marker='o', linestyle='-', label=f"Feature {feature1}")
    plt.plot(x_vals, f2_vals, marker='s', linestyle='--', label=f"Feature {feature2}")

    plt.xlabel("File number")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_with_ratio(folder_path, prefix, feature_main=3, show_ratio=False,
                            title="", ylabel="Value"):
    """
    Plot a main feature from the first line of multiple text files with a given prefix,
    and optionally plot the ratio min(feature_main, feature 4)/max(feature_main, feature 4).
    
    Args:
        folder_path (str): Path to folder containing files
        prefix (str): Filename prefix to select files
        feature_main (int): Index of the main feature to display
        show_ratio (bool): If True, also plot the ratio
        title (str): Plot title
        ylabel (str): Y-axis label
    """
    file_pattern = os.path.join(folder_path, f"{prefix}*.txt")
    file_list = sorted(glob.glob(file_pattern))
    
    if len(file_list) == 0:
        raise ValueError(f"No files found starting with '{prefix}' in {folder_path}.")

    x_vals = []
    main_vals = []
    ratio_vals = []

    for file in file_list:
        basename = os.path.basename(file)
        num_str = ''.join(filter(str.isdigit, basename))
        if not num_str:
            continue
        num = int(num_str)
        x_vals.append(num)

        data = np.loadtxt(file)
        f_main = data[0, feature_main]
        f4 = data[0, 4]  # colonne 4 toujours utilisée pour le ratio
        main_vals.append(f_main)

        ratio = min(f_main, f4) / max(f_main, f4) if max(f_main, f4) != 0 else 0
        ratio_vals.append(ratio)

    # Trier selon les numéros de fichiers
    x_vals, main_vals, ratio_vals = zip(*sorted(zip(x_vals, main_vals, ratio_vals)))

    plt.figure(figsize=(8,5))
    plt.plot(x_vals, main_vals, marker='o', linestyle='-', label=f"Feature {feature_main}")
    
    if show_ratio:
        plt.plot(x_vals, ratio_vals, marker='s', linestyle='--', label="Ratio min/max feature 4")

    plt.xlabel("File number")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    plt.show()

prefix = "methodestat"  # les fichiers commencent par "xxx"
#plot_feature_with_ratio(folder, prefix, feature_main=1, show_ratio=True,title="medium dim", ylabel="Value")
folder = "../resultsreal/mediumdim"
plot_two_features(folder, prefix, feature1=0, feature2=2, title="mediumdim", ylabel="Value")
prefix = "methode"

plot_column_k_files(folder, prefix, file_nums=[14,15,16], column_index=1,title="ARI",
                      ylabel="Value",legend_fontsize=10, line_nums=[0,1,4,6,7,8,9,10],labels=["OUR","DEC","VADE"])

'''
#calcule les moyennes sur les sélectionnés.
m,s =mean_std_by_method(folder, prefix, file_nums=[0,1,2,3,4,5,6,8,10,11,12,13,14,15,16],
                              column_indices=[0,1,2], line_nums=[0,1,4,6,7,8,9,10])

print("m \ns=",m,s)

'''
'''
#oermet de compter les pourcentages!
valid_files, counts, p1, n = count_max_per_file_percent(folder, prefix, 
                                                              file_nums=[0,1,2,3,4,5,6,8,10,11,12,13,14,15,16],
                                                              column_index=0, 
                                                              line_nums=[0,1,4,6,7,8,9,10])
valid_files, counts, p2, n = count_max_per_file_percent(folder, prefix, 
                                                              file_nums=[0,1,2,3,4,5,6,8,10,11,12,13,14,15,16],
                                                              column_index=1, 
                                                              line_nums=[0,1,4,6,7,8,9,10])
valid_files, counts, p3, n = count_max_per_file_percent(folder, prefix, 
                                                              file_nums=[0,1,2,3,4,5,6,8,10,11,12,13,14,15,16],
                                                              column_index=2, 
                                                              line_nums=[0,1,4,6,7,8,9,10])

print(p1)
print(p2)
print(p3)

method_labels = ["KMEANS", None, None, "DBSCAN", None, None, "DPCCE", 
                 None,None, None,"POCS", None, "OUR", "DEC", "VADE"]
print("len",len (p1), len (method_labels))
plot_stacked_bars(p1, p2, p3, labels=method_labels)
'''

'''
folder = "resultsreal/highdim"
plot_column_two_files(folder, prefix, file_nums=[15,14], column_index=1,title="high",
                      ylabel="Value",legend_fontsize=10)
'''
'''
prefix = "methodestat"
folder = "resultsreal/highdim"
plot_two_features(folder, prefix, feature1=0, feature2=2, title="highdim", ylabel="Value")
'''
