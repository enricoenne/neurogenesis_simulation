# Neurogenesis simulation




## Installation and setup

The recommended installation requires Conda (you can install it from [their official website](https://continuumio-docs.readthedocs-hosted.com/miniconda/install/))

### 1. clone the repository

Open Conda Prompt

   ```bash
   git clone https://github.com/enricoenne/neurogenesis_simulation
   cd neurogenesis_simulation
   ```

### 2. create and set up the Conda environemnt

    ```bash
    conda env create -f environment.yml
    conda activate neuro_sim-env
    python -m ipykernel install --user --name neuro_sim-env --display-name "Python (neuro_sim-env)"
    ```

### 3. run the notebooks

    ```bash
    jupyter lab
    ```

It will open a browser window

* select on the left the notebook of intereset (.ipynb format)

    check that the selected kernel (top right) is `Python (neuro_sim-env)`
    

## Notebooks

### data_extraction.ipynb

* Cell cycle length
* Cell cycle exit
* Apical-basal mitosis proportion

### data_extraction_brdu.ipynb

* Extraction of nuclear density and position from BrdU pictures

### data_extraction_layers_from_brdu.ipynb

* Estimation of layer production timeline from BrdU data

### data_extraction_cortex.ipynb

* Estimation of layer neuron population from DAPI data

### results.ipynb

* Simulation with default parameters
* Simulation with higher cell cycle exit rate (Indirect to Direct Neurogenesis)
* Simulation Direct to Indirect Neurogenesis


### results_direct_exploration_high-res.ipynb

* Exploration of parameters compatible with Direct to Indirect Neurogenesis
* Selection of representative parameter combinations

### results_cell-cycle-length.ipynb

* Exploration of different Basal Progenitor cell cycle lengths



## license

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
