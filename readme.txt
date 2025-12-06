Supervised Clustering Framework

A unified evaluation and experimentation pipeline for clustering, supervised clustering, neighborhood-based reasoning and deep baselines.

📌 Overview

This repository contains the full implementation of our supervised clustering framework.
The anaconda environmnent is available in : environment.yml

It includes:

Synthetic dataset generation with controllable difficulty

Real-data processing and evaluation

A neighborhood-based clustering method (our proposed approach)

A large collection of competitor algorithms

Deep-learning baselines (DEC, VADE, etc.) and Deep learning codes ICC, DEPICT and SCAN specific to two image datasets.

A complete experimental pipeline for reproducing all results from the paper

The implementation is modular and fully configurable through command-line arguments or optional YAML configuration.

📂 Repository Structure
├── dataGeneration/              # Synthetic dataset generator
├── concurrents/                 # Implementations of competitor methods
├── ToolsCode/                   # Utility functions (stats, visualization, I/O, etc.)
├── models/                      # Saved and load models
├── realdata/                    # Path for real datasets
├── resultsreal/                 # Results on real datasets
├── resultsynthetic/             # Results on synthetic datasets
├── config.yaml                  # Optional YAML configuration
├── main.py                      # Main entry point (the file you provided)
└── README.md                    # This file


🚀 Features
✔ Synthetic Dataset Generator
Gaussian clusters placed on an n-dimensional hypercube, with:

variable separation
noise injection
anisotropic transformations
random perturbation

✔ Synthetic Dataset Evaluation

Gaussian clusters placed on an n-dimensional hypercube, with:

variable separation
noise injection
anisotropic transformations
random per-feature deviations
adjustable difficulty (“Easy”, “Medium”, “Difficult”)

✔ Real Data Evaluation

Automatic loading in folder, scaling, visualization, and comparison with competitors.

✔ Competitor Methods

The framework includes wrappers for:
K-Means, X-Means, MeanShift, DBSCAN, ADBSCAN, SNN, RNN-DBSCAN, AND-Clust, POCS, DEMUNE, and more.

✔ Deep Baselines for Tabular data

DEC
VADE

✔ Deep Baselines for Image data sets (MNIST and dSprites)
IIC
DEPICT
SCAN

✔ Our Proposed Method

A neighborhood-based supervised clustering method with:

embedding dimension control
fuzzy or crisp patterns
Hamming-based neighborhood selection
tunable hyperparameters

🛠 Installation
git clone https://github.com/<yourname>/<repository>.git


▶ Running Experiments

The main script is:

python main.py


Only one mode can be activated at a time:

Flag	Description
--genpoint_s	generate synthetic pattern dataset
--createmodel	train the neighborhood-based model
--testmodel_synthetic	evaluate on synthetic datasets
--testmodel_real	evaluate on real datasets
--deeplearning_methods	run DEC / VADE baselines specific to two images datasets.
--experiments_synthetic directly call a function with double loops to test synthetic data with any methods, any dimensions... 
--experiments_real      directly call a fonction to handle datasets stored in a directory

parser.add_argument("--experiments_synthetic", type=bool, default=False)
    parser.add_argument("--experiments_real"
Example:

1. Generate training data
python main.py --genpoint_s True

2. Train our model
python main.py --createmodel True --epoch 200

3. Run synthetic experiment
python main.py --testmodel_synthetic True --dimension_test 10

4. Test on real datasets 
python main.py --testmodel_real True

5. Launch deep-learning competitor methods
python main.py --deeplearning_methods True

6. Launch scripts to evaluate synthetic and real sets directly ( in F_examples).
You can directly put parameters you want to use (select the methods, the dimensions, the number of items per dimensions, the amount of noise, the standard deviation...) 

⚙ Configuration

You can either use command-line arguments or a YAML file.

To activate YAML:
useyaml = True
cfg = get_config_yaml("config.yaml")

📊 Outputs

The framework automatically generates:

recognition rates
ARI / NMI / Silhouette statistics

visualizations with TSNE projections

global statistics aggregated across datasets

.txt files summarizing results for each method

All results are stored under:
resultsynthetic/
resultsreal/

