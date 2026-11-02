# Neurogenesis simulation

Simulation of mouse cortical neurogenesis based on

* apical and basal progenitor proliferation

* cell cycle exit (or cell to cell transition probabilities)

* neuronal production

## Preview

Pre-generated plots can be viewed opening directly in the browser the respective notebook (.ipynb) files in the *notebooks* folder.

## System requirements

### Hardware requirements

The simulation code requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements

#### OS

The code has been tested on:
* Windows 11
* Linux Mint 22.1
* macOS Golden Gate 27.0.1

#### other softwares

The recommended installation requires

* Conda: to manage the Python environemnt
    * you can istall the minimal Miniconda software from [their official website](https://continuumio-docs.readthedocs-hosted.com/miniconda/install/) (it could take 5 min)
* git: to clone the repository
    * you can install it from [their official website](https://git-scm.com/install/) (it could take 2 min)


## Installation and setup (up to 10 min)

Open Anaconda Prompt (in Windows) or any terminal with `conda` access and execute the following commands:

### 1. clone the repository

    git clone https://github.com/enricoenne/neurogenesis_simulation
    cd neurogenesis_simulation

### 2. create and set up the Conda environemnt

    conda env create -f environment.yml
    conda activate neuro_sim-env
    python -m ipykernel install --user --name neuro_sim-env --display-name "Python (neuro_sim-env)"

### 3. run the notebooks

    jupyter lab

It will open a browser window

* select on the left the notebook of intereset (.ipynb format) in the **notebooks** folder

* select the right kernel (top right)
    * the default is `Python 3 (opykernel)`
    * select kernel `Python (neuro_sim-env)` instead

Blocks of code running the simulation might require a few minutes to execute (a progress bar is shown)
    

## Notebooks

**data_extraction.ipynb**

* Cell cycle length
* Cell cycle exit
* Apical-basal mitosis proportion

**data_extraction_brdu.ipynb**

* Extraction of nuclear density and position from BrdU pictures

**data_extraction_layers_from_brdu.ipynb**

* Estimation of layer production timeline from BrdU data

**data_extraction_cortex.ipynb**

* Estimation of layer neuron population from DAPI data

**results.ipynb**

* Simulation with default parameters
* Simulation with higher cell cycle exit rate (Indirect to Direct Neurogenesis)
* Simulation Direct to Indirect Neurogenesis

**results_direct_exploration_high-res.ipynb**

* Exploration of parameters compatible with Direct to Indirect Neurogenesis
* Selection of representative parameter combinations

**results_cell-cycle-length.ipynb**

* Exploration of different Basal Progenitor cell cycle lengths

**Theoretical neurogenesis**

* Calculation of optimal neurogenesis curve to achieve right layer neuron distribution

## Simulation code

The simulation code is contained in the simulation_neurog.py file

## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
