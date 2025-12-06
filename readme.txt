# Supervised Clustering Framework

A unified evaluation and experimentation pipeline for clustering, supervised clustering, neighborhood-based reasoning, and deep-learning baselines.

---

## 📌 Overview

This repository contains the full implementation of our supervised clustering framework.  
The Anaconda environment is provided in: `environment.yml`.

It includes:

- **Synthetic dataset generation** with controllable difficulty.
- **Real-data processing and evaluation**.
- **A neighborhood-based clustering method** (our proposed approach).
- **A large collection of competitor algorithms**.
- **Deep-learning baselines** (DEC, VADE, IIC, DEPICT, SCAN) specific to tabular and image datasets.
- **A complete experimental pipeline** for reproducing all results from the paper.
- **Modular implementation** fully configurable via command-line arguments or optional YAML configuration.

---

## 📂 Repository Structure

├── dataGeneration/ # Synthetic dataset generator
├── concurrents/ # Implementations of competitor methods
├── ToolsCode/ # Utility functions (stats, visualization, I/O)
├── models/ # Saved and loaded models
├── realdata/ # Real datasets
├── resultsreal/ # Results on real datasets
├── resultsynthetic/ # Results on synthetic datasets
├── config.yaml # Optional YAML configuration
├── main.py # Main entry point
└── README.md # This file


---

## 🚀 Features

### ✔ Synthetic Dataset Generator
- Gaussian clusters placed on an n-dimensional hypercube.
- Variable separation and noise injection.
- Anisotropic transformations and per-feature perturbations.
- Adjustable difficulty: “Easy”, “Medium”, “Difficult”.

### ✔ Real Data Evaluation
- Automatic loading, scaling, and visualization.
- Comparison with competitor methods.

### ✔ Competitor Methods
Wrappers for:  
`K-Means, X-Means, MeanShift, DBSCAN, ADBSCAN, SNN, RNN-DBSCAN, AND-Clust, POCS, DEMUNE`, and more.

### ✔ Deep Learning Baselines
- **Tabular data**: DEC, VADE  
- **Image datasets (MNIST, dSprites)**: IIC, DEPICT, SCAN

### ✔ Proposed Method
Neighborhood-based supervised clustering with:  
- Embedding dimension control  
- Fuzzy or crisp patterns  
- Hamming-based neighborhood selection  
- Tunable hyperparameters  

---

## 🛠 Installation

```bash
git clone https://github.com/frederic-ros/ParadigmClustering2025.git
conda env create -f environment.yml
conda activate novelPytorch
▶ Running Experiments
The main script:


python main.py
Only one mode can be activated at a time:

Flag	Description
--genpoint_s	Generate synthetic pattern dataset
--createmodel	Train the neighborhood-based model
--testmodel_synthetic	Evaluate on synthetic datasets
--testmodel_real	Evaluate on real datasets
--deeplearning_methods	Run DEC / VADE / image-based baselines
--experiments_synthetic	Run synthetic experiments with custom parameters
--experiments_real	Run experiments on real datasets in a folder

Examples:

Generate training data:

python main.py --genpoint_s True
Train our model:


python main.py --createmodel True --epoch 200
Run synthetic experiment:


python main.py --testmodel_synthetic True --dimension_test 10
Test on real datasets:


python main.py --testmodel_real True
Launch deep-learning competitor methods:


python main.py --deeplearning_methods True
Directly run scripts for custom synthetic/real experiments (in F_examples/).

⚙ Configuration
You can use command-line arguments or a YAML configuration file.

To activate YAML:

useyaml = True
cfg = get_config_yaml("config.yaml")
📊 Outputs
The framework automatically generates:

Recognition rates

ARI / NMI / Silhouette statistics

TSNE visualizations

Global statistics aggregated across datasets

.txt files summarizing results for each method

All results are stored under:

resultsynthetic/
resultsreal/
📌 Notes
Ensure only one mode is active at a time.

For deep-learning baselines, additional dataset preparation may be required.

yaml





