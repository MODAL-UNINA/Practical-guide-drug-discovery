# Practical and Reproducible AI-driven Modeling Protocols in Drug Discovery

This repository contains files associated with the chapter **"Practical and Reproducible AI-driven Modeling Protocols in Drug Discovery."** The materials provided here are intended to support reproducibility and practical implementation of AI techniques in drug discovery.

## Repository Contents

- **Data Preprocessing Script:**  
  The script responsible for cleaning, filtering, and preparing the raw data for further analysis. This script outlines the necessary steps to process the original dataset into a format suitable for AI-driven modeling.

- **Preprocessed Dataset:**  
  The dataset resulting from the data preprocessing pipeline. This file contains the curated and processed data used for developing and evaluating predictive models in drug discovery.

## Original Dataset

The data provided in this repository is based on the Papyrus dataset:

> Béquignon, O.J.M., Bongers, B.J., Jespers, W. et al. Papyrus: a large-scale curated dataset aimed at bioactivity predictions. *Journal of Cheminformatics*, **15**, 3 (2023).  
> [https://doi.org/10.1186/s13321-022-00672-x](https://doi.org/10.1186/s13321-022-00672-x)

## How to Use
**Environment Setup** 

To ensure compatibility and reproducibility, we recommend using Conda to manage dependencies. If you are running this project in Google Colab, you can install the required packages manually.

 1. **Using Conda (Recommended)**
    Using Conda allows you to easily manage dependencies and avoid conflicts.
    
    **Step 1**: Create and activate the environment
    Run the following command to create an isolated Conda environment:
    
    ```bash
    conda create --name qsar_env 
    conda activate qsar_env

    **Step 2**: Install dependencies
    Once the environment is activated, install all required packages from the requirements.txt file:
    
    ```bash
    conda install -c conda-forge --file requirements.txt



2. **Using Google Colab**  
   If you are running the code in Google Colab, install the necessary dependencies manually by executing:

   ```bash
   !pip install rdkit pandas numpy matplotlib seaborn pyscf tensorflow
