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

## Instructions for data analysis code for the paper
Code for producing the figures in the paper can be run by calling the scripts `figX.m` in MATLAB. Data required for 
running the code is available [here](https://doi.org/10.6084/m9.figshare.22493308)

Some jupyter notebooks are provided for data pre-processing:

- `session_average_parameters_per_animal.ipynb` generates the summary parameter fits of session-averaged transition
functions of animals (Fig. 1)
  
- `switching_world_classifier.ipynb` performs the forward simulations that generate results in Fig. 5, S3, S4

- `switching_world_classifier.ipynb` generates the evaluation dataset used to evaluate the classifier performance 
in Fig. 5f.
  
- `src.run_multi_hmm_fits.py` is run to generate the HMM fits for each animal.



