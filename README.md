# Data Analytics Final Project

**Title:**  
**"A Comparative Study of Accuracy and Recommendation Quality in Sequential and Matrix Factorization Methods Across Movie, Book, and Anime Datasets"**

This repository contains the source code and datasets used in the final MSc Data Analytics project.

----------

## Data

All datasets are located in the `dataset/` folder. Due to large file sizes, **each dataset has been sampled to 10% of its original size**. This allows the notebooks to run efficiently, though outputs may differ if rerun with these sampled datasets.

> **Note:** The current notebook outputs reflect results from the original full datasets and are consistent with what has been reported in the final project.

The full original datasets can be downloaded from the following links:
https://drive.google.com/file/d/1lMquwmOHmkccqp5CZDln4VhkT9SbV3Qe/view?usp=sharing

### Dataset Folder Structure

-   `animelist/`: Curated data from MyAnimeList. The data used in the experiments (already sampled and split into train/validation/test sets) is located in `data_sample_split/` subfolder.
    
-   `goodbooks-10k/`: Contains the Goodbooks-10k dataset (user-book ratings from FastML). The processed data for experiments is located in `data_sample_split/` subfolder.
    
-   `ml-1m/`: Includes the MovieLens 1M dataset. The train/validation/test splits used are in `data_split/`.
    
-   `data_robustness_test/`: Contains specially prepared datasets with varying sparsity levels for robustness testing.
    

----------

## Code Overview

Jupyter notebooks follow the naming format:  
`<dataset>_<method/process>.ipynb`  
Where `<dataset>` can be `animelist`, `goodbooks`, or `movielens`.

### Main Notebooks

-   **`<dataset>_eda.ipynb`**  
    Exploratory Data Analysis (EDA) for the dataset. Includes sampling, descriptive statistics, distribution plots, and association rule mining.
    
-   **`<dataset>_matrix_factorization.ipynb`**  
    Implements and evaluates recommendation methods including a Most Popular (MostPop) baseline and matrix factorization algorithms (SVD and BPR), using both default and tuned parameters.
    
-   **`<dataset>_gru.ipynb`**  
    Implementation and evaluation of GRU4Rec with both default and optimized parameters.
    
-   **`<dataset>_matrix_factorization_hyperparameter_opt.ipynb`**  
    Hyperparameter optimization for SVD and BPR using the Optuna library.
    
-   **`<dataset>_gru_hyperparameter_opt.ipynb`**  
    Hyperparameter tuning experiments for GRU4Rec using Optuna.
    

### Additional Notebooks

-   **`all_qualitative_evaluation.ipynb`**  
    Performs qualitative evaluation (coverage, diversity, novelty, serendipity) for MostPop, SVD, BPR, and GRU4Rec across all datasets.
    
-   **`goodbooks_bpr_gru_robustness_test.ipynb`**  
    Analyzes the robustness of BPR and GRU4Rec under cold-start and data sparsity conditions using the Goodbooks dataset.
    

----------

## Requirements

These experiments were conducted in the following environment:

-   **Python**: 3.11.12
    
-   **OS**: Ubuntu 22.04.4 LTS
    
-   **Package Manager**: pip 24.1.2