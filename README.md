This repository contains the code and datasets used in the final MSc Data Analytics project titled:  

**"A Comparative Study of Accuracy and Recommendation Quality in Sequential and Matrix Factorization Methods Across Movie, Book, and Anime Datasets"**.


## Repository Structure

The project includes four main dataset folders:

-   **`animelist/`**  
    Contains curated data from MyAnimeList. The data used in the experiments (already sampled and split into train/validation/test sets) is located in the `data_sample_split/` subfolder.
    
-   **`goodbooks-10k/`**  
    Contains the Goodbooks-10k dataset (user-book ratings from FastML). The processed data for experiments is available in the `data_sample_split/` subfolder.
    
-   **`ml-1m/`**  
    Includes the MovieLens 1M dataset (1 million ratings from ~6,000 users). The train/validation/test split used in experiments is stored in the `data_split/` subfolder.
    
-   **`data_robustness_test/`**  
    Contains specially prepared datasets with different sparsity levels, used for testing model robustness.
    

**Note:**  
To run the qualitative evaluation metrics (coverage, diversity, novelty, serendipity), all prediction results are expected to be saved in the `predictions/` subfolder within each dataset folder (e.g., `animelist/predictions/`).  
Due to large file sizes, these prediction files are not included in the GitHub repository. Download links will be provided separately.  
Additionally, the original full datasets (before filtering and sampling) are also excluded from the repository for the same reason and will be made available via external links.

----------

## Code Overview

Jupyter notebooks are named using the format:  
`<Dataset>_<Method/Process>.ipynb`, where `<Dataset>` is one of: `Animelist`, `Goodbooks`, or `Movielens`.

### Main Processes

-   **EDA**  
    Exploratory data analysis of each dataset.
    
-   **Collaborative Filtering**  
    Experiments using SVD, BPR, and Most Popular (MostPOP) methods with both default and tuned hyperparameters.
    
-   **GRU**  
    Sequential modeling using GRU4Rec, evaluated with default and optimized parameters.
    
-   **Hyperparameter Optimization**  
    Parameter and epoch tuning for SVD and BPR methods.
    
-   **GRU Hyperparameter Optimization**  
    Tuning experiments for GRU4Rec.
    

### Additional Notebooks

-   **`All_Qualitative_Evaluation.ipynb`**  
    Evaluates all datasets and methods using qualitative metrics: coverage, diversity, novelty, and serendipity.
    
-   **`Robustness_to_Sparsity_and_Cold_Start.ipynb`**  
    Tests the impact of data sparsity and cold-start scenarios on BPR and GRU4Rec using the Goodbooks dataset.
    

----------

## Requirements

The experiments were conducted using:

-   **Python**: 3.11.12
    
-   **OS**: Ubuntu 22.04.4 LTS
    
-   **Package Manager**: pip 24.1.2