import glob
import os
import numpy as np
import scipy.io

from src.utils import pathsetup
import smartload.smartload as smart
from exputils import load_raw_info_multiple
import matplotlib.pyplot as plt
import tqdm


def load_raw_choices_outcomes(animal, version):
    '''
    animal, version, version_save: strings
    returns array of rho values for all sessions specified in the fit range file
    raw_choices, raw_outcomes, each is array with size Nblocks x Tmax
    filenames: a list of strings of length Nfiles (same as length of the fit range)
    indicating the names of the files that will be extracted

    '''
    print(f'Starting run for {animal}')
    # Load data
    paths = pathsetup('matchingsim')
    filepath = f"{paths['expdatapath']}/{version}/{animal}_all_sessions_{version}.mat"
    fitrangefile = f"{paths['expdatapath']}/102121/fitranges_122221.mat"

    # changed 3.15.22
    #fitrangefile = f"{paths['expdatapath']}/102121/fitranges_022822_wf.mat"

    if os.path.exists(fitrangefile):
        datarange = smart.loadmat(fitrangefile)
        fitrange = datarange['ranges'][datarange['animals'] == animal][0]
    else:
        raise IOError('File does not exist')

    # Retruns the names of the session
    datasmart = smart.loadmat(filepath)
    filenames = datasmart['session_names'][fitrange]

    raw_choices, raw_outcomes = load_raw_info_multiple(filepath, fitrange)

    return raw_choices, raw_outcomes, filenames

def find_num_errors_leading_block(outcome_block):
    '''
    outcomeblock: a 1d list of outcomes in the block
    with trailing nan's (padded)
    returns an int: number of errors leading the block
    '''
    if outcome_block[0] == 1: #flip on first run
        return 0 #TODO: should be -1 or 0?
    else:
        cands = np.where(outcome_block == 1)[0]
        if len(cands) == 0: # fail to flip for the whole block
            return np.sum(~np.isnan(outcome_block))
        else:
            return cands[0]


def find_num_rewards_trailing_block(outcome_block):
    '''
    outcomeblock: a 1d list of outcomes in the block
    with trailing nan's (padded)
    returns an int: number of rewards trailing the block
    '''
    outcome_block = outcome_block[~np.isnan(outcome_block)]
    assert(len(outcome_block) > 0)
    outcome_block = np.flip(outcome_block)

    if outcome_block[0] == 0: #all errors in this block, no rewards
        return 0 #TODO: should be -1 or 0?
    else:
        cands = np.where(outcome_block == 0)[0]
        if len(cands) == 0: # all block rewarded
            return np.sum(~np.isnan(outcome_block))
        else:
            return cands[0]

def fit_rho_multi_sessions(outcomes_raw, verbose=0):
    '''
    outcomes_raw: a list of arrays of length Nsess, each of size Nblocks x Tmax
    (with nan's padded)
    returns: rhos, an array of size Nblocks, giving the rho value for
    each of the session
    Ne_lst: a list of length Nsess, each containing an (Nblocks - 1) x 1 array of
    the number of errors leading the blocks
    Nr_lst: a list of length Nsess, each containing an (Nblocks - 1) x 1 array of
    the number of rewards trailing the blocks
    '''
    rhos = []
    Ne_lst = []
    Nr_lst = []
    for idx, item in enumerate(outcomes_raw):
        if verbose:
            print(idx)
        rhos.append(fit_rho_single_session(item)[0])
        Ne_lst.append(fit_rho_single_session(item)[1])
        Nr_lst.append(fit_rho_single_session(item)[2])
    return rhos, Ne_lst, Nr_lst

def fit_rho_single_session(outcomelst):
    '''
    outcomelst: an array of size Nblocks x Tmax (with nan's padded)
    returns: the fitted rho value of the session
    '''
    Nblocks = outcomelst.shape[0]

    Ne = []
    Nr = []
    for i in range(Nblocks - 1):
        # find the number of errors leading block i + 1
        Ne.append(find_num_errors_leading_block(outcomelst[i + 1,:]))

        # Find the number of rewards trailing block i
        Nr.append(find_num_rewards_trailing_block(outcomelst[i,:]))

    if Nblocks <= 5:
        return np.nan, Ne, Nr# too few blocks to calculate rho..

    # rho is correlation of Ne/Nr
    return np.corrcoef(Ne, Nr)[0, 1], Ne, Nr



if __name__ == '__main__':
    version = '122221b'
    version_save = '122221b'
    paths = pathsetup('matchingsim')
    files = glob.glob(f"{paths['expdatapath']}/{version}/*_all_sessions_{version}.mat")

    # Get the animal name from the file list
    animals = [file.split('/')[-1].split('_')[0] for file in files]

    num_states = 6
    N_iters = 3000

    showplots = 0
    savefile = 1


    for animal in tqdm.tqdm(animals):
        raw_choices, raw_outcomes, fnames = load_raw_choices_outcomes(animal, version)
        rhos, Ne_lst, Nr_lst = fit_rho_multi_sessions(raw_outcomes)

        if showplots:
            plt.figure()
            plt.plot(rhos)
            plt.show()

        if savefile:
            savefilename = f"{paths['datapath']}/rho_data/{version_save}/{animal}_rhofit_{version_save}.mat"
            if not os.path.exists(savefilename):
                scipy.io.savemat(savefilename, dict(rhos=rhos, Ne_lst=Ne_lst, Nr_lst=Nr_lst, fnames=fnames))
                print('file saved!')
            else:
                print('file exists, skipping save!')

        if showplots:
            print(rhos)
