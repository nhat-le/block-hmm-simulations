from src.run_simulations import *
from sklearn import svm
from src.utils import in_percentile



def get_Qmetrics(gamma, eps, N_iters=50, rlow=0):
    '''
    gamma: List[float], learning rate of agents
    eps: List[float], exploration rates of agents
    N_iters: int, number of blocks to simulate
    rlow: float, 0 <= rlow <= 0.5, reward rate of lower side
    Simulate a Q-learning agent for N_iters trials,
    for each trial the behavior is carried out for nblocks (20)
    returns the behavioral metrics (efficiency, lapse, offset, slope)
    averaged across all the blocks in each trial
    '''
    gammalst = [gamma]
    epslst = [eps]

    agent_type = 'qlearning'  # 'qlearning' or 'inf-based' or 'v-accumulation'

    num_states = 2
    obs_dim = 1
    nblocks = 20  # 100
    eps = 0
    hmm_fit = False
    sigmoid_window = 30
    ntrials_per_block = [15, 25]
    seed = 0
    # rlow = 0
    rhigh = 1 - rlow
    np.random.seed(123)

    params = {'N_iters': N_iters, 'num_states': num_states, 'obs_dim': obs_dim,
              'nblocks': nblocks, 'eps': eps, 'ntrials_per_block': ntrials_per_block,
              'gammalst': gammalst, 'epslst': epslst, 'seed': seed, 'type': agent_type, 'hmm_fit': hmm_fit,
              'seed': seed, 'sigmoid_window': sigmoid_window, 'rlow': rlow, 'rhigh': rhigh, 'verbose': False}

    # Q-learning simulation
    #     print('Starting q-learning simulation')
    efflistQ, T11lstQ, T22lstQ, E1lstQ, E2lstQ, PRslopelistQ, PLslopelistQ, \
    PRoffsetlistQ, PLoffsetlistQ, LapseLQ, LapseRQ, ParamsAQ, ParamsBQ, ParamsCQ = run_repeated_single_agent(params)

    Qmetrics = [efflistQ, LapseLQ, PLoffsetlistQ, PLslopelistQ]
    return Qmetrics


def get_IB_metrics(psw, prew, N_iters=50, rlow=0):
    '''
       psw: List[float], prob. of switching states
       prew: List[float], prob. of reward (0.5 <= prew < 1) of the high side
       N_iters: int, number of blocks to simulate
       rlow: float, 0 <= rlow <= 0.5, reward rate of lower side
       Simulate a Q-learning agent for N_iters trials,
       for each trial the behavior is carried out for nblocks (20)
       returns the behavioral metrics (efficiency, lapse, offset, slope)
       averaged across all the blocks in each trial
   '''
    pswitchlst = [psw]
    prewlst = [prew]

    agent_type = 'inf-based'  # 'qlearning' or 'inf-based' or 'v-accumulation'

    num_states = 2
    obs_dim = 1
    nblocks = 20  # 100
    eps = 0
    hmm_fit = False
    sigmoid_window = 30
    ntrials_per_block = [15, 25]
    seed = 0
    # rlow = 0
    rhigh = 1 - rlow
    np.random.seed(123)

    params = {'N_iters': N_iters, 'num_states': num_states, 'obs_dim': obs_dim,
              'nblocks': nblocks, 'eps': eps, 'ntrials_per_block': ntrials_per_block,
              'seed': seed, 'type': agent_type, 'hmm_fit': hmm_fit,
              'pswitchlst': pswitchlst, 'prewlst': prewlst,
              'seed': seed, 'sigmoid_window': sigmoid_window, 'rlow': rlow, 'rhigh': rhigh, 'verbose': False}

    # Inference-based simulation
    efflistIB, T11lstIB, T22lstIB, E1lstIB, E2lstIB, PRslopelistIB, PLslopelistIB, \
    PRoffsetlistIB, PLoffsetlistIB, LapseLIB, LapseRIB, ParamsAIB, ParamsBIB, ParamsCIB = run_repeated_single_agent(
        params)

    IBmetrics = [efflistIB, LapseLIB, PLoffsetlistIB, PLslopelistIB]
    return IBmetrics


def fit_and_evaluate_svm(Qmetrics, IBmetrics):
    # Make the training and test sets
    XQ = np.vstack(Qmetrics)
    XIB = np.vstack(IBmetrics)
    Xall = np.hstack([XQ, XIB])
    yall = np.array([0] * XQ.shape[1] + [1] * XIB.shape[1])

    # Normalize
    mean = np.mean(Xall, axis=1)
    std = np.std(Xall, axis=1)
    Xall = (Xall.T - mean) / std

    # Shuffle
    order = np.arange(Xall.shape[0])
    np.random.shuffle(order)
    Xall = Xall[order, :]

    Xtrain = Xall[:75, :]
    Xtest = Xall[75:, :]
    ytrain = yall[order][:75]
    ytest = yall[order][75:]

    clf = svm.SVC(kernel='linear')
    clf.fit(Xtrain, ytrain)
    coefs = clf.coef_

    preds = clf.predict(Xtest)
    perf = sum(preds == ytest) / len(preds)
    return perf, coefs


def find_svm_perf(gamma, eps, psw, prew):
    pswitchlst = [psw]
    prewlst = [prew]

    gammalst = [gamma]
    epslst = [eps]

    agent_type = 'qlearning'  # 'qlearning' or 'inf-based' or 'v-accumulation'

    N_iters = 50
    num_states = 2
    obs_dim = 1
    nblocks = 20  # 100
    eps = 0
    hmm_fit = False
    sigmoid_window = 30
    ntrials_per_block = [10, 40]
    seed = 0
    rlow = 0
    rhigh = 1
    np.random.seed(123)

    params = {'N_iters': N_iters, 'num_states': num_states, 'obs_dim': obs_dim,
              'nblocks': nblocks, 'eps': eps, 'ntrials_per_block': ntrials_per_block,
              'gammalst': gammalst, 'epslst': epslst, 'seed': seed, 'type': agent_type, 'hmm_fit': hmm_fit,
              'pswitchlst': pswitchlst, 'prewlst': prewlst,
              'seed': seed, 'sigmoid_window': sigmoid_window, 'rlow': rlow, 'rhigh': rhigh, 'verbose' :False}

    # Q-learning simulation
    print('Starting q-learning simulation')
    efflistQ, T11lstQ, T22lstQ, E1lstQ, E2lstQ, PRslopelistQ, PLslopelistQ, \
    PRoffsetlistQ, PLoffsetlistQ, LapseLQ, LapseRQ, ParamsAQ, ParamsBQ, ParamsCQ = run_repeated_single_agent(params)

    # Inference-based simulation
    print('Starting inf-based simulation')
    params['type'] = 'inf-based'
    efflistIB, T11lstIB, T22lstIB, E1lstIB, E2lstIB, PRslopelistIB, PLslopelistIB, \
    PRoffsetlistIB, PLoffsetlistIB, LapseLIB, LapseRIB, ParamsAIB, ParamsBIB, ParamsCIB = run_repeated_single_agent \
        (params)

    Qmetrics = [efflistQ, LapseLQ, PLoffsetlistQ, PLslopelistQ]
    IBmetrics = [efflistIB, LapseLIB, PLoffsetlistIB, PLslopelistIB]
    perf, coefs = fit_and_evaluate_svm(Qmetrics, IBmetrics)
    return Qmetrics, IBmetrics, perf, coefs


def normalize_and_average(arr, average=True):
    normarr = (arr - np.median(arr)) / np.std(arr)
    if average:
        return np.mean(normarr, axis=2)
    else:
        return normarr

def normalize_and_average_all(lst, average=True):
    return [normalize_and_average(elem, average) for elem in lst]


def find_Q_IB_ave_distance(expmetrics, QIBmetrics):
    # Normalize the metrics
    expeff_norm = (expmetrics[0] - np.median(QIBmetrics[0])) / np.std(QIBmetrics[0])
    explapse_norm = (expmetrics[1][2] - np.median(QIBmetrics[1])) / np.std(QIBmetrics[1])
    expoffset_norm = (expmetrics[1][1] - np.median(QIBmetrics[2])) / np.std(QIBmetrics[2])
    expslope_norm = (expmetrics[1][0] - np.median(QIBmetrics[3])) / np.std(QIBmetrics[3])

    QIBnorm = normalize_and_average_all(QIBmetrics)
    print(QIBnorm[0].shape)
    Qeff_norm = QIBnorm[0][:11, :]
    Qlapse_norm = QIBnorm[1][:11, :]
    Qslope_norm = QIBnorm[3][:11, :]
    Qoffset_norm = QIBnorm[2][:11, :]
    IBeff_norm = QIBnorm[0][11:, :]
    IBlapse_norm = QIBnorm[1][11:, :]
    IBslope_norm = QIBnorm[3][11:, :]
    IBoffset_norm = QIBnorm[2][11:, :]

    Qdistance = (Qeff_norm - expeff_norm) ** 2 + (Qlapse_norm - explapse_norm) ** 2 + \
                (Qslope_norm - expslope_norm) ** 2 + (Qoffset_norm - expoffset_norm) ** 2
    IBdistance = (IBeff_norm - expeff_norm) ** 2 + (IBlapse_norm - explapse_norm) ** 2 + \
                 (IBslope_norm - expslope_norm) ** 2 + (IBoffset_norm - expoffset_norm) ** 2

    return Qdistance, IBdistance, expeff_norm, explapse_norm, expoffset_norm, expslope_norm



def find_Q_IB_zdistance(expmetrics, QIBmetrics, method='z'):
    '''
    Find the distance by first z-scoring according to the distribution of
    each parameter combination
    expmetrics: (eff, [slope, offset, lapse]) tuple
    QIBmetrics: [eff, lapse, offset, slope] array
    '''

    if method == 'z': # z-score method
        eff_norm = (expmetrics[0] - np.nanmean(QIBmetrics[0], axis=2)) / (np.nanstd(QIBmetrics[0], axis=2) + 1e-4)
        lapse_norm = (expmetrics[1][2] - np.nanmean(QIBmetrics[1], axis=2)) / (np.nanstd(QIBmetrics[1], axis=2) + 1e-4)
        offset_norm = (expmetrics[1][1] - np.nanmean(QIBmetrics[2], axis=2)) / (np.nanstd(QIBmetrics[2], axis=2) + 1e-4)
        slope_norm = (expmetrics[1][0] - np.nanmean(QIBmetrics[3], axis=2)) / (np.nanstd(QIBmetrics[3], axis=2) + 1e-4)
    elif method == 'percentile': # percentile method, aiming for '50 percentile'
        eff_norm = in_percentile(QIBmetrics[0], expmetrics[0]) - 0.5
        lapse_norm = in_percentile(QIBmetrics[1], expmetrics[1][2]) - 0.5
        offset_norm = in_percentile(QIBmetrics[2], expmetrics[1][1]) - 0.5
        slope_norm = in_percentile(QIBmetrics[3], expmetrics[1][0]) - 0.5

    distance = eff_norm**2 + lapse_norm**2 + offset_norm**2 + slope_norm**2

    return distance[:11, :], distance[11:, :], eff_norm, lapse_norm, offset_norm, slope_norm


