import autograd.numpy as np
import autograd.numpy.random as npr
import os
import scipy.io
import numpy as np2
npr.seed(0)

import ssm
import smartload.smartload as smart
from src.exputils import load_multiple_sessions, make_savedict
npr.seed(0)

def run_and_validate(animal, seed, params):
    print(f'Starting run and save for {animal}, seed {seed}')
    # Load data
    # version = '_113021'
    # version_save = '_113021'
    version = params['version']
    filepath = f'/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/expdata/{version}/{animal}_all_sessions_{version}.mat'
    fitrangefile = params['fitrangefile']
    # fitrangefile = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/expdata/102121/fitranges_102121.mat'
    datarange = smart.loadmat(fitrangefile)
    fitrange = datarange['ranges'][datarange['animals'] == animal][0]
    obs, lengths, dirs, fnames, rawchoices, _ = load_multiple_sessions(filepath, fitrange, trialsperblock=15)


    # Run the fitting procedure
    nstates_lst = params['nstates_lst']
    N_iters = params['N_iters']
    frac_train = params['frac_train']

    # Build the train and test sets
    ntrials, obs_dim = obs.shape
    ntrain = int(ntrials * frac_train)

    np.random.seed(seed)
    masks = ~np.isnan(obs)
    obsmasked = obs[:]
    obsmasked[~masks] = 1

    order = np2.random.permutation(ntrials)
    obs_train = obs[np.sort(order[:ntrain]), :]
    obs_test = obs[np.sort(order[ntrain:]), :]

    # Get the baseline performance (of a Bernoulli model)
    p_bernoulli = np.sum(obs_train) / (np.shape(obs_train)[0] * np.shape(obs_train)[1])
    obs_test_flat = obs_test.flatten()
    ber_LLH = obs_test_flat * np.log(p_bernoulli) + (1 - obs_test_flat) * np.log(1 - p_bernoulli)
    L0 = sum(ber_LLH)


    ll_lst = []
    for num_states in nstates_lst:
        hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
        ll_test, obs_train, obs_test = run_hmm_lls(hmm, obs_train, obs_test, masks, N_iters, L0)
        ll_lst.append(ll_test)
        print(f'Num states = {num_states}, likelihood = {ll_test}')

    print(nstates_lst)
    print(ll_lst)

    return ll_lst, nstates_lst, obs_train, obs_test

def run_and_validate_synthetic(obs, seed, params):
    # Run the fitting procedure
    nstates_lst = params['nstates_lst']
    N_iters = params['N_iters']
    frac_train = params['frac_train']

    # Build the train and test sets
    ntrials, obs_dim = obs.shape
    ntrain = int(ntrials * frac_train)

    np.random.seed(seed)

    obs_train = obs[:ntrain, :]
    obs_test = obs[ntrain:, :]

    # Get the baseline performance (of a Bernoulli model)
    p_bernoulli = np.sum(obs_train) / (np.shape(obs_train)[0] * np.shape(obs_train)[1])
    obs_test_flat = obs_test.flatten()
    ber_LLH = obs_test_flat * np.log(p_bernoulli) + (1 - obs_test_flat) * np.log(1 - p_bernoulli)
    L0 = sum(ber_LLH)

    masks = None


    ll_lst = []
    for num_states in nstates_lst:
        hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
        ll_test, obs_train, obs_test = run_hmm_lls(hmm, obs_train, obs_test, masks, N_iters, L0)
        ll_lst.append(ll_test)
        print(f'Num states = {num_states}, likelihood = {ll_test}')

    print(nstates_lst)
    print(ll_lst)

    return ll_lst, nstates_lst, obs_train, obs_test

def run_hmm_lls(hmm, obs_train, obs_test, masks, N_iters, L0):

    hmm.fit(obs_train, method="em", masks=masks, num_iters=N_iters, init_method="kmeans")

    # Evaluate the hmm on the test set
    ll_test = (hmm.log_likelihood(obs_test) - L0) / obs_test.shape[0] / np.log(2)

    return ll_test, obs_train, obs_test




def run_animal(animal, seeds):
    '''
    Given animal name and seed number, run block HMM over all the seeds and report the results
    :param animal: str: animal name
    :param seeds: list[int], seed names
    :return: None
    '''

    # Load data
    version = '_113021'
    filepath = f'/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/{animal}_all_sessions{version}.mat'
    fitrangefile = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/fitranges_102121.mat'
    datarange = smart.loadmat(fitrangefile)
    fitrange = datarange['ranges'][datarange['animals'] == animal][0]
    obs, lengths, dirs, fnames, rawchoices = load_multiple_sessions(filepath, fitrange, trialsperblock=15)

    # Run the fitting procedure
    N_iters = 500
    obs_dim = obs.shape[1]
    num_states = 4

    lls_all = []
    for seed in seeds:
        np.random.seed(seed)
        masks = ~np.isnan(obs)
        obsmasked = obs[:]
        obsmasked[~masks] = 1

        hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
        hmm_lls = hmm.fit(obs, method="em", masks=masks, num_iters=N_iters, init_method="kmeans")

        lls_all.append(hmm_lls[-1])
        print(f'animal {animal}, seed value = {seed}, hmm LLS = {hmm_lls[-1]:.2f}')

    # Determine the best seed
    idbest = np2.argmax(lls_all)
    print(f'Best seed is: {seeds[idbest]}')
    return seeds[idbest]

# if __name__ == '__main__':
#     seeds = [121, 122, 123, 124, 125]
#     # animals = ['e46']
#     # animals = ['f02', 'f03', 'f04', 'f11', 'f12', 'e35', 'e40',
#     #     'fh01', 'fh02', 'f05', 'e53', 'fh03', 'f16', 'f17', 'f20', 'f21', 'f22', 'f23']
#     # animals = ['e53', 'fh03', 'f16', 'f17', 'f20', 'f21', 'f22', 'f23']
#     animals = ['f01']
#     params = dict(nstates_lst=np.arange(2, 8),
#                   N_iters=500,
#                   frac_train=0.8)
#     for animal in animals:
#         try:
#             seed = 123
#             # Run and save with the best seed
#             run_and_validate(animal, seed, params)
#         except:
#             continue

if __name__ == '__main__':
    ### CODE FOR EXP FIT EVALUATIONS
    # Setup evaluation parameters
    nstates_lst = np.arange(1, 9)
    N_iters = 3000
    frac_train = 0.8

    params = dict(nstates_lst=np.arange(1, 9),
                  N_iters=3000,
                  frac_train=0.8,
                  version='122221b',
                  fitrangefile='/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/expdata/102121/fitranges_102121.mat')

    # params = dict(nstates_lst=[1], N_iters=10, frac_train=0.8)

    animal_lst = ['f01']  # , 'f02', 'f03', 'f04', 'f11', 'f12', 'e35', 'e40',
    #         'fh01', 'fh02', 'e53', 'fh03', 'f16', 'f17', 'f20', 'f21', 'f22', 'f23']

    ll_lst_all = []
    test_lens = []
    for animal in animal_lst:
        print(f'Analyzing animal {animal}')
        ll_lst, nstates_lst, obs_train, obs_test = run_and_validate(animal, 123, params)
        test_lens.append(obs_test.shape[0])
        ll_lst_all.append(ll_lst)



    ### CODE FOR SYNTHETIC SIMULATIONS
    #Create the dataset
    # Set the parameters of the HMM
    # time_bins = 5000  # number of time bins
    # num_states = 3  # number of discrete states
    # obs_dim = 30  # dimensionality of observation
    #
    # # Make an HMM
    # gen_seed = 125
    # np.random.seed(gen_seed)
    # true_hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
    #
    # true_hmm.observations.mus = np.array([1, 9, 4]).T
    # true_hmm.observations.sigmas = np.array([0.8, 1.5, 0.2]).T
    # true_hmm.observations.lapses = np.array([0.15, 0.05, 0.3]).T
    #
    # # true_hmm.transitions.transition_matrix = np.array([[0.98692759, 0.01307241],
    # #                                        [0.00685383, 0.99314617]])
    #
    # # Sample some data from the HMM
    # true_states, obs = true_hmm.sample(time_bins)
    #
    # print(obs.shape)
    #
    # # Setup evaluation parameters
    # nstates_lst = np.arange(1, 8)
    # N_iters = 3000
    # frac_train = 0.8
    #
    # ll_lst, nstates_lst, obs_train, obs_test = run_and_validate_synthetic(obs, seed=123, params=dict(nstates_lst=nstates_lst,
    #                                                                                                  N_iters=N_iters, frac_train=frac_train))