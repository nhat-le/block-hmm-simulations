### General information
Code tested on MAC OS 10.15.7, MATLAB 2020a, and Python 3.9

## Instructions for running the analysis and simulation scripts

### Basic setup
0. Install MATLAB, python, anaconda and jupyter notebook if these have not been installed
1. Clone the directory `git clone https://github.com/nhat-le/switching-simulations`
2. Add the folder `MatchingSimulations/PaperFigures/code/helpers` to the MATLAB path
3. In the script `.../code/helpers/pathsetup.m`, modify the following directories:
   * `out.rigboxpath` should point to the directory where the rigbox behavioral datafiles are stored
   * `out.rootpath` should point to the root directory (the full path to the `MatchingSimulations` folder)
4. Install the following Python packages: `smartload`: `python -m pip install [package_name]`
5. Install the following fork of the `ssm` package: `https://github.com/nhat-le/ssm` (Note: modified from original package `https://github.com/lindermanlab/ssm`)


### 0. Experimental schematics
* Example animal behavior (Fig. 1C): run `PaperFigures/code/schematic/fig1c.m`
* Example behavior of model-free and inference-based agent (Fig. 1D): first run the code block in the section `Fig. 1d` in the notebook `switching_world_characterization.ipynb` to generate the behavior of q-learning or inference-based agents (specifying the parameters in the `params` dictionary, these parameters will be saved in `processed_data/simdata/schematic/qlearning-sim-122221.mat`). 

    + Then run `PaperFigures/code/schematic/fig1d.m`
  The data for the schematics are located in `processed_data/simdata/schematic`
      
### 1. Previous analytical approaches
* Generation of `rho` schematics (Fig. 2A-D): first run the notebook `fig2abcd.ipynb`. The notebook will save simulated data to `PaperFigures/code/schematic/data`. Then run `PaperFigures/code/schematic/fig2abcd.m` to plot the figures.

* Generation of logistic regression fits (Fig. 2E-G): first run the notebook: `fig2efg.ipynb`. The notebook will save simulated data to `processed_data/simdata/pitfalls`. Then run `PaperFigures/code/history/fig2efg.m`

### 2. Characterization of Q-learning and inference-based parameter spaces

* Illustration of behavior metrics s, alpha, epsilon, E (fig. 3A): run `PaperFigures/code/schematic/fig3a.m`

* Landscape of the four behavioral benchmarks (Fig. 3B + 4A): run the code block `Simulating behavior in the parameter space (Fig. 3b + Fig. 4a)` in the notebook `switching_world_characterization.ipynb`.
  The notebook calls the functions `run_simulations.run_multiple_agents`
  Adjust the cell in this notebook which says
  ``agent_type = 'qlearning' ``. The type can be `'qlearning'` or `'inf-based'`
  +  Results are saved in the files named `EGreedyqlearningAgent-withCorr-doublesigmoid-prob0.00to1.00-{version}.mat` in the folder `processed_data/simdata/{version}`

  + To generate the plots, run `code/characterization/fig3b.m` or `code/characterization/fig4a.m`.

  
* Note on fitting procedure: sigmoidal curves are with reference to t = 0 at the first trial in the block. For example, if offset = 1, this means that at trial 2, P(correct) = 0.5

#### Notable parameters:
```
- N_iters = 50 
- num_states = 2
- obs_dim = 1
- nblocks = 1000 #important! sometimes set to 20
- sigmoid_window = 30
- ntrials_per_block = [10, 40]
```

* Example Q-learning and inference-based agents (Fig. 3C,D + 4B): run the code block `Simulation of example q-learning and inf-based agents over multiple blocks (Fig. 3c,d + Fig. 4b)` in the notebook `switching_world_characterization.ipynb`.

  + Then run `PaperFigures/code/characterization/fig3cd_4b.m.m`
  
Current versions of the figures are based on: version `121021`

Note that: `PLoffsetlist` is negative, `PLslopelist` is positive in the saved .mat files


### 3. Behavioral regime segmentation with the watershed algorithm 

* Behavioral regime segmentation with the tsne algorithm (Fig. 5 + Fig. S3):
Common parameters are defined in `makeparams_tsne.m`. This includes parameters such as perplexity (for TSNE), as well as parameters for the density-based segmentation.

  + Run script `PaperFigures/code/characterization/fig5.m`. Should specify the data version (currently using `092321`). Note that the plots generated are: behavioral regime demarcation, tsne visualization, transition function visualization, and options/outputs are saved if specified.
  
  + Note: change `opts.prob` to 1, 0.9, 0.8, 0.7 for different types of environments.

### 4. Decoding analysis (Fig. 6)
* Feature summary (Fig. 6a): run `PaperFigures/code/characterization/fig6a.m`

* Training and evaluating decoders on synthetic data (Fig. 6C-E):
  - First run the notebook `switching_world_classifier.ipynb` to generate the synthetic data. Change `rlow` to the right probability of reward (for e.g. `rlow = 0.1` corresponds to a 90-10 world)

  - Results will be saved in files with the form `decodingresults_from_pickle_092221_prob0.00.mat`, etc.

  - Fig. 6C,D: Run `PaperFigures/code/decoding/fig6cd.m` to train decoders. Decoders saved in `processed_data/decoding/models`.
    
  - Fig. 6E: run the script `PaperFigures/code/decoding/fig6e.m`. Need to change the `opts.decodingdir` and `opts.decoding_version` to the file with the correct date. This is the file where the processed data (Python simulations) are stored (usually in the `processed_data/decoding` folder)

### 5. Single-session fitting and decoding (Fig. 7)

* Evolution of behavioral features (Fig. 7a): first run the notebook `fig7.ipynb` to pre-process animal data. This will save the data into `expdata/version/fitparams_session_averaged_{version}.mat`.
    - Then, run `code/expfit/fig7a.m`

* Decoding of session-averaged behavioral mode (Fig. 7b,c): run `PaperFigures/code/expfit/fig7bc.m`

* Decoding in probabilistic sessions (Fig. 7d): run `PaperFigures/code/expfit/fig7d.m`


### 6. BlockHMM on synthetic data (Fig. 8)

* Simple behavior session (Fig. 8A): `PaperFigures/code/schematic/fig8a.m`

* To generate the synthetic data used in Fig. 8, run `blockHMM_simulations.ipynb`.
  
Raw data files will be saved in the folder `processed_data/simdata/blockhmm`. This also includes K-selection on synthetic data.

  - Then run `PaperFigures/code/blockhmm/fig8.m`.

### 7. BlockHMM on experimental data (Fig. 9)

* First, run script `PaperFigures/code/blockhmm/compile_behavioral_sessions.m` to produce _all_sessions files

* K-selection by cross-validation (Fig. S4): Run `blockHMM_K_selection.ipynb`. Results are saved in `processed_data/blockhmmfit/K_selection`.
  - Then run `code/blockhmm/figS4.m`

* Note: the `fitrange` file, located in `processed_data/expdata/102121/fitranges_122221.mat`, contains the ranges of files that are good to analyze. For example, if `fitrange` is [6,7,8,9], the blockhmm protocol only looks at the sessions 6 to 9 (note these are Python zero-indexed) in the saved `_all_sessions` file

* Pre-processing steps:
  - Run `src/run_multi_hmm_fits.py`. Remember to change `save` option if you want the fitted results to be saved.
  - The blockhmm fit results will be saved in the folder `processed_data/blockhmmfit/{expdate}`
  - Parameters to change:
     - `num_states`: K value (number of clusters)
     - `version` (line 18): version number of the files in `expdata` that is loaded
     - `version_save`  (line 19): version number of the file that is saved (in `blockhmmfit`)
     - `fitrangefile` (line 25): path to the fitrange file which specifies the range of files to fit
     - `savefile` (line 66): whether to save the results

  - Run `code/blockhmm/hmmblockstate_preprocess.m`, change the `expfitdate` to different folders for different K values. For e.g. `121821bK3` contains only the fit files for animals with K = 3.

* Fig. 9 + Fig. S5,6: run `code/blockhmm/fig9.m`



### Supplementary figures
* Fig.S1 and S2: run `PaperFigures/code/characterization/figS1S2.m`
* FigS7: run `src/run_multi_rho_fits.py`, then `PaperFigures/coe/expfit/figS7.m`



