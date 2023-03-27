import scipy.io
import autograd.numpy as np
from ssm.util import split_by_trials
import ssm
import os.path
import smartload.smartload as smart
from tqdm.notebook import tqdm

def sigmoid(x, mu, sigma, lapse=0):
    return lapse + (1-2*lapse) / (1 + np.exp(-(x - mu) * sigma))

def get_id_range(filepath):
    '''
    identify the first session that has data in the mat file
    '''
    id = 0
    res = None
    while res == None:
        res = load_session(filepath, id)
        if res == None:
            id += 1

    data = scipy.io.loadmat(filepath)
    idend = len(data['choices_cell'][0])

    return np.arange(id, idend)


def load_raw_info_multiple(filepath, idlst):
    '''
    Inputs: filepath: a path to the raw data file of a single animal
    Returns choices, outcomes: two lists containing the choices and
    outcomes of all the session in the fitrange file
    '''
    choices_all = [load_raw_info_single(filepath, id)[0] for id in idlst]
    outcomes_all = [load_raw_info_single(filepath, id)[1] for id in idlst]
    return choices_all, outcomes_all


def load_raw_info_single(filepath, id):
    '''
    Inputs: filepath: a path to the raw data file of a single animal
    id: id of file to read in the list from the choice structure
    returns: choices, outcomes: the raw choices and outcomes of the session
    '''
    data = scipy.io.loadmat(filepath)

    if len(data['choices_cell'][0][id]) == 0:
        print('empty file:', id)
        return None

    choices = data['choices_cell'][0][id][0].astype('float')
    outcomes = data['feedbacks_cell'][0][id][0].astype('float')
    targets = data['targets_cell'][0][id][0].astype('float')

    bpos = np.where(np.diff(targets))
    bpos = np.hstack([-1, bpos[0], len(targets) - 1])
    blens = np.diff((bpos))

    choicearr = split_by_trials(choices, blens, chop='max')
    outcomesarr = split_by_trials(outcomes, blens, chop='max')

    return choicearr, outcomesarr



def load_session(filepath, id, trialsperblock=15):
    data = scipy.io.loadmat(filepath)

    # print(id)

    if len(data['choices_cell'][0][id]) == 0:
        print('empty file:', id)
        return None
    choices = data['choices_cell'][0][id][0].astype('float')
    targets = data['targets_cell'][0][id][0].astype('float')

    if 'opto_cell' in data:
        opto = data['opto_cell'][0][id][0].astype('float')


    # flip choices for targets = 0
    signedtargets = 1 - 2 * targets
    signedchoices = (choices * signedtargets + 1) / 2

    bpos = np.where(np.diff(targets))
    bpos = np.hstack([-1, bpos[0], len(targets) - 1])
    blens = np.diff((bpos))

    choicearr = split_by_trials(signedchoices, blens, chop='max')[:, :trialsperblock]
    rawchoice = split_by_trials(signedchoices, blens, chop='max')

    if 'opto_cell' in data:
        opto = split_by_trials(opto, blens, chop='max')
    else:
        opto = np.nan

    blocktargets = targets[bpos[1:]]
    # print(f'id = {id}, shape: {choicearr.shape}')

    if choicearr.shape[1] != trialsperblock:
        # invalid session, just retun array of nan's
        choicearr = np.ones((1, trialsperblock)) * np.nan
        blocktargets = [np.nan]

    # TODO: Decide what to do with nan's that are returned from split_by_trials
    # print(np.sum(np.isnan(choicearr)))

    return choicearr, blocktargets, rawchoice, opto


def load_multiple_sessions(filepath, idlst, trialsperblock=15):
    '''
    filepath: path to extracted data .mat file
    idlst: list of sessions to extract
    returns: concatenated choice, num trials of sessions, and names of sessions
    '''
    choicearrs = [load_session(filepath, id, trialsperblock)[0] for id in tqdm(idlst)]
    blocktargets = [load_session(filepath, id, trialsperblock)[1] for id in tqdm(idlst)]
    rawchoices = [load_session(filepath, id, trialsperblock)[2] for id in tqdm(idlst)]
    opto = [load_session(filepath, id, trialsperblock)[3] for id in tqdm(idlst)]
    blocktargets = np.hstack(blocktargets)

    # return session names
    datasmart = smart.loadmat(filepath)
    filenames = datasmart['session_names'][idlst]

    return np.vstack(choicearrs), [arr.shape[0] for arr in choicearrs], blocktargets, filenames, rawchoices, opto


def make_savedict(vars, builtin_names):
    '''
    Make a dictionary of variables to save
    :param vars: a list of variable names
    :return: a dict of var -> value mappings
    '''
    collection = [entry for entry in builtin_names.items() if entry[0] in vars]
    # print(len(collection))
    assert len(collection) == len(vars)
    return dict(collection)


def run_and_save(animal, savefilename, savefile=1):
    # TODOs:
    # -Handle nan choices
    # - Faster optimization and avoid local minima?

    # Load data
    filepath = f'/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/{animal}_all_sessions.mat'
    # filepath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/f01_all_sessions.mat'
    fitrange = get_id_range(filepath)
    obs, lengths, dirs = load_multiple_sessions(filepath, fitrange)

    masks = ~np.isnan(obs)
    obsmasked = obs[:]
    obsmasked[~masks] = 0

    N_iters = 500
    obs_dim = obs.shape[1]
    num_states = 4

    # Fit the HMM
    hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
    hmm.fit(obsmasked, method="em", masks=masks, num_iters=N_iters, init_method="kmeans")

    # Save the result
    zstates = hmm.most_likely_states(obs)
    transmat = hmm.transitions.transition_matrix
    params = hmm.observations.params

    vars = ['zstates', 'dirs', 'lengths', 'transmat', 'params', 'fitrange', 'filepath', 'obs']
    savedict = make_savedict(vars, locals())

    savepathroot = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/'
    savepath = savepathroot + savefilename

    if savefile and not os.path.exists(savepath):
        scipy.io.savemat(savepath, savedict)
        print('File saved')
    elif os.path.exists(savepath):
        print('File exists, skipping save..')












