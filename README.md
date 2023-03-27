### General information
Code tested on MAC OS 10.15.7, MATLAB 2020a, and Python 3.9

## Instructions for running the analysis and simulation scripts

### Basic setup
0. Install MATLAB, python, anaconda and jupyter notebook if these have not been installed
1. Clone the directory `git clone https://github.com/nhat-le/switching-simulations`
2. Add the folder `block-hmm-simulations` to the MATLAB path
3. Install the following Python packages: `smartload`: `python -m pip install [package_name]`
4. Install the following fork of the `ssm` package: `https://github.com/nhat-le/ssm` (Note: modified from original package `https://github.com/lindermanlab/ssm`)

### Running blockHMM on synthetic data

* To generate and fit the synthetic data, run `blockHMM_simulations.ipynb`.
  
Raw data files will be saved in the file `blockhmm_simulated.mat`. 
Data for the K-selection will be saved in `blockhmm_synthetic_K_validation.mat`.

* Then run the MATLAB script `blockhmm_synthetic.m` to generate the figures.


