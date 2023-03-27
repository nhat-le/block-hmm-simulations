from src.worldModels import *
import scipy.optimize
import ssm


def pathsetup(project):
    out = {}
    if project == 'matchingsim':
        out['datapath'] = '/Users/minhnhatle/Dropbox (MIT)/Sur/conferences_and_manuscripts/DynamicForagingPaper/NatureCommunications/Submission 1/code/processed_data/'
        out['codepath'] = '/Users/minhnhatle/Dropbox (MIT)/Sur/conferences_and_manuscripts/DynamicForagingPaper/NatureCommunications/Submission 1/code/PaperFigures/code/'
        out['blockhmmsimdata'] ='/Users/minhnhatle/Dropbox (MIT)/Sur/conferences_and_manuscripts/DynamicForagingPaper/NatureCommunications/Submission 1/code/processed_data/simdata/blockhmm'

        out['expdatapath'] = out['datapath']+ 'expdata'
        out['blockhmmfitpath'] = out['datapath']+ 'blockhmmfit'
        out['simdatapath'] = out['datapath']+ 'simdata'
        out['decodingdatapath'] = out['datapath']+ 'decoding'
        out['decodingconfigpath'] = out['datapath']+ 'decoding/configs'
        out['decodingmodelpath'] = out['datapath']+ 'decoding/models'

        out['blockhmm_codepath'] = out['codepath']+ 'blockhmm'
        out['characterize_codepath'] = out['codepath']+ 'characterization'
        out['decoding_codepath'] = out['codepath']+ 'decoding'
        out['expfit_codepath'] = out['codepath']+ 'expfit'
        out['schematic_codepath'] = out['codepath']+ 'schematic'
        out['Kselection_path'] = out['datapath'] + 'blockhmmfit/K_selection'

    elif project == 'tca':
        out['datapath'] = '/Users/minhnhatle/Documents/ExternalCode/tca/data'
        out['codepath'] = '/Users/minhnhatle/Documents/ExternalCode/tca/src/matlab'
        out['rawdatapath'] = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/raw/extracted'
        out['tcamatpath'] = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors'

    else:
        raise ValueError('invalid project, must be tca or matchingsim')

    return out

def logistic(x, beta=1):
    return 1 / (1 + np.exp(-beta * x))


def make_switching_world(rlow, rhigh, nblocks, ntrialsLow, ntrialsHigh):
    ratesL = (np.mod(np.arange(nblocks), 2)) * (rhigh - rlow) + rlow
    ratesR = (1 - np.mod(np.arange(nblocks), 2)) * (rhigh - rlow) + rlow
    if np.random.rand() > 0.5:
        rates = np.vstack((ratesL, ratesR)).T
    else:
        rates = np.vstack((ratesR, ratesL)).T
    ntrials = np.random.uniform(low=ntrialsLow, high=ntrialsHigh, size=nblocks).astype('int')
    world = PersistentWorld(rates=rates, ntrials=ntrials)
    return world, ntrials


def make_switching_world_withCheck(rlow, rhigh, nblocks, ntrialsLow, ntrialsHigh):
    ratesL = (np.mod(np.arange(nblocks), 2)) * (rhigh - rlow) + rlow
    ratesR = (1 - np.mod(np.arange(nblocks), 2)) * (rhigh - rlow) + rlow
    if np.random.rand() > 0.5:
        rates = np.vstack((ratesL, ratesR)).T
    else:
        rates = np.vstack((ratesR, ratesL)).T
    ntrials = np.random.uniform(low=ntrialsLow, high=ntrialsHigh, size=nblocks).astype('int')
    world = PersistentWorldWithCheck(rates=rates, ntrials=ntrials, threshold=0.8)
    return world, ntrials



def find_LR_transition_fit(world, agent, window, type='sigmoid'):
    '''
    For a switching world, determines the agent transition functions,
    for left->right and right->left transitions
    window: how many trials after the transition do we want to keep for fitting?
    '''
    # Get the choices around the transition
    if world.ntrialblocks[-1] == 0:
        choicelst = split_by_trials(np.array(agent.choice_history), world.ntrialblocks[:-1], chop='max')
    else:
        choicelst = split_by_trials(np.array(agent.choice_history), world.ntrialblocks, chop='max')

    if window is None:
        window = choicelst.shape[1]
    choicelst = choicelst[:, :window]


    if np.ndim(np.array(world.side_history)) == 1:
        print('here, first side = ', )
        pRight, pLeft = fit_sigmoidal(choicelst, first_side=world.side_history[0])
    else:
        # print('here 2, first side = ', world.side_history[0][0])
        if type == 'sigmoid':
            pRight, pLeft = fit_sigmoidal(choicelst, first_side=world.rate_history[0][0] < 0.5)
        elif type == 'doublesigmoid':
            odd_rows = choicelst[1::2,:]
            even_rows = choicelst[::2,:]

            len_odds = max(np.sum(~np.isnan(odd_rows), axis=1))
            len_evens = max(np.sum(~np.isnan(even_rows), axis=1))

            odd_rows = odd_rows[:,:len_odds]
            even_rows = even_rows[:,:len_evens]

            # TODO: confirm side is correct!
            if world.rate_history[0][0] > 0.5:
                # print('left')
                leftAverage = np.nanmean(odd_rows, axis=0)
                rightAverage = np.nanmean(even_rows, axis=0)
            else:
                # print('right')
                rightAverage = np.nanmean(odd_rows, axis=0)
                leftAverage = np.nanmean(even_rows, axis=0)

            pfit = fit_doublesigmoid_helper(leftAverage, rightAverage)
            [offsetL, slopeL, offsetR, slopeR, lapseL, lapseR] = pfit
            pRight = [slopeR, offsetR, lapseR]
            pLeft = [slopeL, offsetL, lapseL]
    return pRight, pLeft, choicelst


def find_experiment_metrics(data, window, type='sigmoid'):
    '''
    Determine the parameter of the block switching in a given behavioral session
    filename: string, name of behavior .mat file containing allchoices and alltargets
    window: how many trials after the transition do we want to keep for fitting?
    '''
    # data = scipy.io.loadmat(filename)

    blocktrans = np.where(np.diff(data['alltargets']))[1]
    blocksizes = np.diff(blocktrans)
    blocksizes = np.hstack([blocktrans[0] + 1, blocksizes])
    # print(blocksizes)

    choicelst = split_by_trials(data['allchoices'][0, :sum(blocksizes)], blocksizes, chop='max')

    if window is None:
        window = choicelst.shape[1]
    choicelst = choicelst[:, :window]

    choicelst = (choicelst + 1) / 2

    if type == 'sigmoid':
        pRight, pLeft = fit_sigmoidal(choicelst, first_side=1-data['alltargets'][0][0])
    elif type == 'doublesigmoid':
        if data['alltargets'][0][0] != 0:
            # print('left')
            leftAverage = np.nanmean(choicelst[1::2, :], axis=0)
            rightAverage = np.nanmean(choicelst[::2, :], axis=0)
        else:
            # print('right')
            rightAverage = np.nanmean(choicelst[1::2, :], axis=0)
            leftAverage = np.nanmean(choicelst[::2, :], axis=0)

        pfit = fit_doublesigmoid_helper(leftAverage, rightAverage)
        [offsetL, slopeL, offsetR, slopeR, lapseL, lapseR] = pfit
        pRight = [slopeR, offsetR, lapseR]
        pLeft = [slopeL, offsetL, lapseL]

    elif type == 'exponential':
        first_side = 1-data['alltargets'][0][0]
        pRight, _ = fit_expfun2([1, 0], np.arange(window), 1-np.mean(choicelst[first_side::2,:], axis=0)) #fit_exponential(choicelst, first_side=1 - data['alltargets'][0][0])
        pLeft, _ = fit_expfun2([1, 0], np.arange(window), np.mean(choicelst[1-first_side::2,:], axis=0)) #fit_exponential(choicelst, first_side=1 - data['alltargets'][0][0])

    # Efficiency
    feedback = (-data['allchoices']) == (data['alltargets'].astype('int') * 2 - 1)
    eff = np.sum(feedback) / len(feedback[0])

    return pRight, pLeft, choicelst, eff



def split_by_trials(seq, ntrials, chop='none'):
    '''
    seq: an array of integers, of length sum(ntrials)
    ntrials: number of trials per block
    splits the seq into small chunks with lengths specified by ntrials
    chop: if none, no chopping, if min, chop to the shortest length (min(ntrials)),
    if max, pad to the longest length (min(ntrials)),
    '''
    assert(len(seq) == sum(ntrials))
    if ntrials[-1] == 0:
        ntrials = ntrials[:-1]

    minN = min(ntrials)
    maxN = max(ntrials)
    if len(seq) != sum(ntrials):
        raise ValueError('ntrials must sum up to length of sequence')
    endpoints = np.cumsum(ntrials)[:-1]

    splits = np.split(seq, endpoints)

    if chop == 'min':
        # print('here')
        # print(np.array([elem[:minN] for elem in splits]))
        return np.array([elem[:minN] for elem in splits])

    elif chop == 'none':
        return splits
    elif chop == 'max':
        # pad to the max len
        result = np.ones((len(ntrials), maxN)) * np.nan
        for i in range(len(ntrials)):
            result[i, 0:ntrials[i]] = splits[i]
        return result
    else:
        raise ValueError('invalide chop type')


def get_zstates(agent, num_states=2, obs_dim=1, N_iters=50):
    '''
    Fit HMM to the choice sequence of the agent
    Returns: the sequence of the most likely z-states
    '''
    # Fit HMM to choice sequence
    data = np.array(agent.choice_history)[:, None]

    ## testing the constrained transitions class
    hmm = ssm.HMM(num_states, obs_dim, observations="bernoulli")
    hmm_lls = hmm.fit(data, method="em", num_iters=N_iters, init_method="kmeans", verbose=0)

    if hmm.observations.logit_ps[0] > 0:
        return 1 - hmm.most_likely_states(data)
    else:
        return hmm.most_likely_states(data)


def get_switch_times(world, agent):
    '''
    Returns an array of switching times (in trials),
    based on the HMM model fits
    '''
    z_states = get_zstates(agent)
    splits = split_by_trials(z_states, world.ntrialblocks, chop='none')
    # Identify where the switch happens
    if np.ndim(np.array(world.side_history)) == 1:
        first_side = world.side_history[0]
    else:
        first_side = world.side_history[0][1]

    switchlst = []
    for i in range(len(splits)):
        arr = splits[i]
        # print(arr)
        # print(i, first_side)
        # Skip trials that start on the wrong side
        if arr[0] == (first_side + i) % 2:
            switch = -1
            # print('skipping')
        else:
            # Find the first element that is the opposite state
            target = (i + first_side) % 2
            cands = np.where(arr == target)[0]
            if len(cands) == 0:
                switch = world.ntrialblocks[i]
            else:
                switch = cands[0]
        # print('switch =', switch)
        switchlst.append(switch)

    return np.array(switchlst)


def get_num_errors_leading_block(world, agent):
    '''
    Returns an array of switching times (in trials),
    switch times based on the first time animal switches
    '''
    choicelst = split_by_trials(agent.choice_history, world.ntrialblocks, chop='none')

    if np.ndim(np.array(world.side_history)) == 1:
        first_side = world.side_history[0]
    else:
        first_side = 1 - world.side_history[0][0]

    switchlst = []
    for i in range(len(world.ntrialblocks) - 1):
        blockchoice = choicelst[i]
        target = (first_side + i) % 2
        if blockchoice[0] == target: #already switched on the first trial
            switch = -1
        elif np.sum(blockchoice == target) == 0: #no switch happened in the block
            switch = len(blockchoice)
        else:
            switch = np.where(blockchoice == target)[0][0]
        # print(switch)
        switchlst.append(switch)

    return np.array(switchlst)


def get_num_rewards_trailing_block(world, agent):
    '''
    Returns an array of consec rewards at the end of the block
    switch times based on the first time animal switches
    '''
    choicelst = split_by_trials(agent.choice_history, world.ntrialblocks, chop='none')

    if np.ndim(np.array(world.side_history)) == 1:
        first_side = world.side_history[0]
    else:
        first_side = 1 - world.side_history[0][0]

    nrewlst = []
    for i in range(len(world.ntrialblocks) - 1):
        blockchoice = choicelst[i]
        blockchoiceflip = np.flip(blockchoice)
        target = (first_side + i) % 2
        if blockchoiceflip[0] != target:
            nrew = -1
        else:
            nrew = np.where(blockchoiceflip != target)[0]
            if len(nrew) == 0:
                nrew = len(blockchoiceflip)
            else:
                nrew = nrew[0]

        nrewlst.append(nrew)

    return np.array(nrewlst)

#### ERROR FUNCTIONS USED FOR CURVE FITTING #####
def predict_sigmoid(x, p):
    '''
    x: an np array of x values
    p: an array of [slope(positive), offset (negative), lapse (0 to 0.5)]
    '''
    return p[2] + (1 - 2 * p[2]) * 1 / (1 + np.exp(-p[0] * (x + p[1])))

def predict_exponential(p, x):
    C = p[1]
    alpha = p[0]
    return C - C * np.exp(-alpha * x)

def predict_doublesigmoid(x, p):
    gamma = p[2]
    lamb = p[3]
    return gamma + (1 - gamma - lamb) * 1 / (1 + np.exp(-p[0] * (x + p[1])))

def error_exponential(p, x, y):
    '''
    Error function used for exponential fitting
    '''
    preds = predict_exponential(p, x)
    return np.sum((preds - y) ** 2)

def errorsigmoid(p, x, y):
    '''
    Error function used for sigmoid fitting
    '''
    # lapse = p[2]
    # preds = lapse + (1 - 2 * lapse) * 1 / (1 + np.exp(-p[0] * (x + p[1])))
    preds = predict_sigmoid(x, p)

    return np.sum((preds - y) ** 2)

def errordoublesigmoid(p, xR, yR, xL, yL):
    '''
    p is an array with [offsetL, slopeL, offsetR, slopeR, lapseL, lapseR]
    '''
    [offsetL, slopeL, offsetR, slopeR, lapseL, lapseR] = p
    predR = predict_doublesigmoid(xR, [slopeR, offsetR, lapseL, lapseR])
    predL = predict_doublesigmoid(xL, [slopeL, offsetL, lapseR, lapseL])
    # print(predL, predR)

    return np.nansum((predL - yL) ** 2) + np.nansum((predR - yR) ** 2) + 0.01 * (slopeL**2 + slopeR**2)

def exp_fun(x, params):
    alpha = params[0]
    beta = params[1]
    C = params[2]
    return C - beta * np.exp(-alpha * x)


def loss(params, x, y):
    pred = exp_fun(x, params)
    return np.sum((pred - y) ** 2)


def exp_fun2(x, params):
    alpha = params[0]
    C = params[1]
    return C - C * np.exp(-alpha * x)


def loss2(params, x, y):
    pred = exp_fun2(x, params)
    return np.sum((pred - y) ** 2)


def find_transition_guess(sig):
    '''
    Returns the offset where a time series crosses 0.5
    '''
    return np.argmin((sig - 0.5) ** 2)


def find_transition_guess_binary(sig):
    '''
    Returns the offset where a time series crosses 0.5, through binary segmentation
    '''
    candidates = np.where(np.diff(sig > 0.5) != 0)[0]
    if sum(sig > 0.5) == 0:
        return 1
    elif len(candidates) == 0:
        return 1
    else:
        return np.where(np.diff(sig > 0.5) != 0)[0][0]


### FITTING FUNCITONS ####
def fit_doublesigmoid_helper(leftAverage, rightAverage):
    '''
    Fit two sigmoids to the left/right average data
    leftAverage is increasing from 0 to 1
    rightAverage is decreasing from 1 to 0
    '''
    offsetsR = np.arange(len(rightAverage))
    offsetsL = np.arange(len(leftAverage))
    funR = lambda x: errordoublesigmoid(x, offsetsR, 1-rightAverage, offsetsL, leftAverage)
    switchGuessR = find_transition_guess_binary(leftAverage)  # offset that crosses 0.5
    switchGuessL = find_transition_guess_binary(rightAverage)
    if sum(leftAverage > 0.5) == 0:
        switchGuessL = len(leftAverage)

    if sum(rightAverage < 0.5) == 0:
        switchGuessR = len(rightAverage)
    # if switchGuessR == -1:  # No switch happened!
    #     switchGuessR = len(rightAverage)
    #
    # if switchGuessL == -1:# No switch happened!
    #     switchGuessL = len(leftAverage)
        # pRight = [0, -np.inf, 0]
    paramsFit = scipy.optimize.minimize(funR, [-switchGuessR, 1, -switchGuessL, 1, 0, 0],
                                          bounds=((None, 0), (0, None), (None, 0), (0, None), (0, 0.5), (0, 0.5)))

    return paramsFit.x


def fit_sigmoidal(choicelst, first_side):
    '''
    Fit a sigmoidal to the average choice data
    first_side: first side that is rewarded, i.e. world.side_history[0][0]
    returns: pright, pleft, where each is a tuple (slope, offset, lapse)
    '''
    # print('choicemean = ', np.mean(choicelst[:,::2]), 'side=  ', first_side)
    if first_side == 0:
        # print('left')
        leftAverage = np.nanmean(choicelst[1::2, :], axis=0)
        rightAverage = np.nanmean(choicelst[::2, :], axis=0)
    else:
        # print('right')
        rightAverage = np.nanmean(choicelst[1::2, :], axis=0)
        leftAverage = np.nanmean(choicelst[::2, :], axis=0)

    offsetsR = np.arange(len(rightAverage))
    offsetsL = np.arange(len(leftAverage))

    # Fit right transitions
    # print(rightAverage)
    funR = lambda x: errorsigmoid(x, offsetsR, rightAverage)
    switchGuessR = find_transition_guess_binary(rightAverage)  # offset that crosses 0.5
    if switchGuessR == -1:  # No switch happened!
        # pRight = [0, -np.inf, 0]
        paramsRight = scipy.optimize.minimize(funR, [1, -len(rightAverage), 0],
                                              bounds=((None, 0), (None, 0), (0, 0.2)))
    else:
        paramsRight = scipy.optimize.minimize(funR, [1, -switchGuessR, 0],
                                              bounds=((None, 0), (None, 0), (0, 0.2)))
    pRight = paramsRight.x
    # print(pRight)

    # print('done with right')
    # Fit left transitions
    # print(leftAverage)
    funL = lambda x: errorsigmoid(x, offsetsL, leftAverage)
    switchGuessL = find_transition_guess_binary(leftAverage)
    if switchGuessL == -1:  # No switch happened!
        # pLeft = [0, -np.inf, 0]
        paramsLeft = scipy.optimize.minimize(funL, [-1, -len(leftAverage), 0],
                                             bounds=((0, None), (None, 0), (0, 0.2)))
    else:
        # print('here')
        paramsLeft = scipy.optimize.minimize(funL, [-1, -switchGuessL, 0],
                                             bounds=((0, None), (None, 0), (0, 0.2)))
    pLeft = paramsLeft.x
    # print(pLeft)
    # print('done with left')

    return pRight, pLeft

def fit_expfun2(params0, datax, datay):
    # Filter out nan's in datax and datay
    goody = datay[~np.isnan(datax) & ~np.isnan(datay)]
    goodx = datax[~np.isnan(datax)& ~np.isnan(datay)]
    # print(goodx, goody)

    result = scipy.optimize.minimize(loss2, params0, (goodx, goody),
                                     bounds=((0, None), (0, None)))
    params = result.x
    ypred = exp_fun2(datax, params)
    return params, ypred


def fit_expfun(params0, datax, datay):
    # Filter out nan's in datax and datay
    goody = datay[~np.isnan(datax)]
    goodx = datax[~np.isnan(datax)]

    result = scipy.optimize.minimize(loss, params0, (goodx, goody),
                                     bounds=((0, None), (0, None), (None, None)))
    params = result.x
    ypred = exp_fun(datax, params)
    return params, ypred

def simulate_rew_error_correlations(world, agent):
    exp = Experiment(agent, world)
    exp.run()

    lst = get_switch_times(world, agent).astype('float')
    # print(lst)
    lst[lst == -1] = np.nan
    nafterswitch = world.ntrialblocks[:-1] - lst

    # Aggregate data
    xarr = nafterswitch[:-1]
    yarr = lst[1:]
    order = np.argsort(xarr)
    xsorted = xarr[order]
    ysortbyX = yarr[order]

    xvals, idx = np.unique(xsorted, return_index=True)
    # print(xsorted)
    ysplit = np.split(ysortbyX, idx[1:])
    # print(ysplit)

    # Mean of each split
    means = []
    stds = []
    for elem in ysplit[1:]:
        # print(elem)
        if sum(~np.isnan(np.array(elem))) == 0: #everything is nan
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(np.nanmean(elem))
            stds.append(np.nanstd(elem) / np.sqrt(len(elem)))

    return xvals[1:], means, stds, ysplit


### USEFUL HELPER FUNCTIONS ###
def in_percentile(arr, val):
    '''
    Returns the percentile of data point 'val' in lst
    '''
    arr = np.array(arr)
    l = arr.shape[-1]
    return np.sum(arr <= val, axis=arr.ndim - 1) / l


def pad_to_same_length(arrlst):
    arrlens = [len(arr) for arr in arrlst]
    maxlen = np.max(arrlens)
    # print(maxlen)
    # pad everything to maxlen
    padlst = []
    for arr in arrlst:
        Ntopad = maxlen - len(arr)
        # print(Ntopad)
        padded = np.pad(np.array(arr, dtype='float'), (0, Ntopad), constant_values=np.nan)
        padlst.append(padded)
    return np.array(padlst)


if __name__ == '__main__':
    pass