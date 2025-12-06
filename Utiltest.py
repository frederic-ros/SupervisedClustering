# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 08:51:55 2025

@author: frederic.ros
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_results_by_dimension(folder_path, dev=0.3, 
                              mask=None,ylabel="", title=""):
    """
    Plot performance curves across dimensions for all files of the form 'result{dim}d{dev}'.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing result files.
    dev : float
        The 'dev' value used in file names (e.g., 0.3 for 'result3d0.3').
    mask : list[bool]
        Boolean vector (length 6) to select which of the 6 metrics to plot.
    """
    if mask is None:
        mask = [True] * 6  # by default plot all

    # Convert dev to string with same format as filename
    dev_str = str(dev)

    # Collect result files matching the pattern
    files = [f for f in os.listdir(folder_path) if f.startswith("result") and f"d{dev_str}" in f]
    if not files:
        print(f"No files found for dev={dev}")
        return

    # Extract dimension from filename: result<dim>d<dev>
    def extract_dim(filename):
        # remove "result" prefix and split by "d"
        try:
            dim_str = filename.replace("result", "").split("d")[0]
            return int(dim_str)
        except:
            return -1

    files = sorted(files, key=extract_dim)

    dims = []
    values = []

    for f in files:
        path = os.path.join(folder_path, f)
        data = np.loadtxt(path)
        if data.ndim > 1:
            data = data.flatten()
        dims.append(extract_dim(f))
        values.append(data[:6])  # take first 6 metrics

    values = np.array(values)

    plt.figure(figsize=(8, 5))
    for i in range(6):
        if mask[i]:
            plt.plot(dims, values[:, i], marker='o', label=f"Metric {i+1}")

    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return values
'''
def plot_comparison(values1, values2, label1="test1", label2="test2",mask=[True,False], ylabel="", title=""):
    """
    Plot comparison of a single metric from two sets of results.

    Parameters
    ----------
    values1, values2 : np.array
        Arrays of shape (n_dims, n_metrics).
    mask : list of bool
        Boolean list of length 6 with only one True to select the metric.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    """
    if mask.count(True) != 1:
        raise ValueError("Mask must have exactly one True value.")

    metric_index = mask.index(True)
    n_dims = values1.shape[0]
    dims = np.arange(n_dims)  # assumes values1 and values2 have same dims

    plt.figure(figsize=(8, 5))
    plt.plot(dims, values1[:, metric_index], marker='o', label=label1)
    plt.plot(dims, values2[:, metric_index], marker='s', label=label2)
    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
'''
def plot_comparison(values1, values2, label1="test1", label2="test2",
                    mask=[True, False], ylabel="", title="", ymin=None):
    """
    Plot comparison of a single metric from two sets of results.

    Parameters
    ----------
    values1, values2 : np.array
        Arrays of shape (n_dims, n_metrics).
    mask : list of bool
        Boolean list of length 6 with only one True to select the metric.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    ymin : float or None
        If set, defines the minimum value of the y-axis (e.g., 0.65).
    """
    if mask.count(True) != 1:
        raise ValueError("Mask must have exactly one True value.")

    metric_index = mask.index(True)
    n_dims = values1.shape[0]
    dims = np.arange(n_dims)  # assumes values1 and values2 have same dims

    plt.figure(figsize=(8, 5))
    plt.plot(dims, values1[:, metric_index], marker='o', label=label1)
    plt.plot(dims, values2[:, metric_index], marker='s', label=label2)
    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Option pour fixer le début de l’axe des ordonnées
    if ymin is not None:
        ymax = max(values1[:, metric_index].max(), values2[:, metric_index].max())
        plt.ylim(ymin, ymax + 0.02 * (ymax - ymin))

    plt.tight_layout()
    plt.show()


def plot_comparison_gen(values_list, labels, mask, ylabel="", title="", 
                        ymin=None, ymax=None, legend_fontsize=9):
    """
    Plot comparison of a selected metric across multiple result sets.

    Parameters
    ----------
    values_list : list of np.array
        List of arrays (each of shape (n_dims, n_metrics)) representing results.
    labels : list of str
        List of labels for each result set (same length as values_list).
    mask : list of bool
        Boolean list (length = n_metrics) with one True to select the metric.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    ymin, ymax : float, optional
        Limits for y-axis.
    legend_fontsize : int, optional
        Font size for the legend (default=9).
    """
    if len(values_list) != len(labels):
        raise ValueError("values_list and labels must have the same length.")
    if mask.count(True) != 1:
        raise ValueError("Mask must have exactly one True value.")

    metric_index = mask.index(True)
    n_dims = values_list[0].shape[0]
    dims = np.arange(n_dims)  # x-axis (assuming same dimensions across all)

    plt.figure(figsize=(8, 5))

    for values, label in zip(values_list, labels):
        plt.plot(dims, values[:, metric_index], marker='o', label=label)

    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Performances via les dimensions
#Mutual, ARI, SIL,%,minmax,cl,% recognition
# Exemple : tracer seulement les courbes 1, 2, 5 et 6
mask = [False, False, False,True,False, False]

values1 = plot_results_by_dimension("resultsynthetique/our/Testnoise/noise0.2_0.3hamming0.01", title="% of recognition score",dev=0.2, mask=mask)
values2 = plot_results_by_dimension("resultsynthetique/our/Testnoise/Noise40percent", title="% of recognition score",dev=0.4, mask=mask)
plot_comparison_gen(
        values_list=[values1, values2],
        labels=["noise=20%", "noise=40%"],
        mask=[False, False, False, False, False, True],
        ylabel="Score",
        title="silhouette score",
        ymin=0.0, legend_fontsize=12)

'''
values1 = plot_results_by_dimension("resultsynthetique/our/Testdimension", title="% of recognition score",dev=0.3, mask=mask)
values2 = plot_results_by_dimension("resultsynthetique/our/Testdimension", title="% of recognition score",dev=0.4, mask=mask)
plot_comparison_gen(
        values_list=[values1, values2],
        labels=["dev=0.3", "dev=0.4"],
        mask=[False, True, False, False, False, False],
        ylabel="Score",
        title="Silhouette score (noise = 20%)",
        ymin=0.3, legend_fontsize=12)
'''

'''
#bruit 20%
values1 = plot_results_by_dimension("resultsynthetique/our/Testnoise/noise0.2_0.3hamming0.01", dev=0.2, mask=mask)
values2 = plot_results_by_dimension("resultsynthetique/our/Testnoise/noise0.2_0.4hamming0.01", 
                          dev=0.2, mask=mask, ylabel="CVI",title="CVI vs dimension noise")

plot_comparison_gen(
        values_list=[values1, values2],
        labels=["dev=0.3", "dev=0.4"],
        mask=[False, False, False, False, True, False],
        ylabel="Score",
        title="minmax % recognition",
        ymin=0.6, legend_fontsize=12)

'''

'''
#bruit.
values1 = plot_results_by_dimension("resultsynthetique/our/Testnoise/noise0.2hamming0.1", dev=0.05, mask=mask)
values2 = plot_results_by_dimension("resultsynthetique/our/Testnoise/noise0.2hamming0.1", dev=0.2, mask=mask)
plot_comparison_gen(values_list=[values1, values2],labels=["noise=5%", "noise=20%"],
                    mask=[True, False, False, False, False, False], 
                    ylabel="Score", 
                    title="% of cluster recognition",ymin=0.8,legend_fontsize=12)
'''

'''
#ABLATION MUTUAL
values1 = plot_results_by_dimension("resultsynthetique/ablation/mutualY", 
                          dev=0.2, mask=mask, ylabel="CVI",title="% recognition")
values2 = plot_results_by_dimension("resultsynthetique/ablation/mutualN", 
                          dev=0.2, mask=mask, ylabel="CVI",title="% recognition")

mask = [False, False, False, False, True, False]  # par exemple, 3ᵉ métrique
plot_comparison(values1, values2, label1="mutual", label2="without mutual",mask=mask, ylabel="Score", title="Comparison of % recognition")
'''
'''
values1 = plot_results_by_dimension("resultsynthetique/our/Testdensity/noise01", dev=0.1, mask=mask)
values2 = plot_results_by_dimension("resultsynthetique/our/Testdensity/noise02", dev=0.2, mask=mask)
plot_comparison(values1, values2, label1="noise=10%", label2="noise=20%",mask=mask, ylabel="Score", 
                title="Adjusted Rand index",ymin=0.0)
'''
'''
#test des models.
values1 = plot_results_by_dimension("resultsynthetique/our/Testnoise/noise0.2_0.3hamming0.01", dev=0.2, mask=mask)
values2 = plot_results_by_dimension("resultsynthetique/our/testmodelHardnoise", dev=0.2, mask=mask)
values3 = plot_results_by_dimension("resultsynthetique/our/testmodelHardsansnoise", dev=0.2, mask=mask)

plot_comparison_gen(
        values_list=[values1, values2, values3],
        labels=["full model", "without membership","without noise"],
        mask=[False, False, False, False, True, False],
        ylabel="Score",
        title="minmax % recognition",
        ymin=0.4, legend_fontsize=12)
'''        