from src.utils import simulate_rew_error_correlations, fit_expfun2, simulate_rew_error_correlations
import src.utils
from src.worldModels import *
from src.agents import *
import numpy as np
import ssm
import tqdm
import statsmodels.api as sm
# from tqdm.notebook import tqdm



def run_rew_error_simulations(params):
    nblocks = params['nblocks']
    # seed = params['seed']
    pstruct = params['pstruct']
    agenttype = params['agenttype']
    prew_world = params['world_pr']
    psw_world = params['world_ps']

    # np.random.seed(seed)
    world = ForagingWorld(prew=prew_world, psw=psw_world, pstruct=pstruct, nblockmax=nblocks)

    if agenttype == 'inf-based':
        prew = params['prew']
        psw = params['psw']
        agent = EGreedyInferenceBasedAgent(prew=prew, pswitch=psw)
    elif agenttype == 'qlearning':
        gamma = params['gamma']
        eps = params['eps']
        agent = EGreedyQLearningAgent(gamma=gamma, eps=eps)
    else:
        raise ValueError('Invalid agent')


    exp = Experiment(agent, world)
    exp.run()

    Nerrors = src.utils.get_num_errors_leading_block(world, agent)[1:]
    Nrews = src.utils.get_num_rewards_trailing_block(world, agent)[:-1]

    Nerrors_trim = Nerrors[(Nerrors > 0) & (Nrews > 0) & (Nrews <= 15)]
    Nrews_trim = Nrews[(Nerrors > 0) & (Nrews > 0) & (Nrews <= 15)]

    return Nerrors_trim, Nrews_trim


def rew_err_sweep(params):
    seed = params['seed']
    np.random.seed(seed)
    agenttype = params['agenttype']
    if agenttype == 'qlearning':
        xlst = params['gamma']
        ylst = params['eps']
        xlabel = 'gamma'
        ylabel = 'eps'
    elif agenttype == 'inf-based':
        xlst = params['psw']
        ylst = params['prew']
        xlabel = 'psw'
        ylabel = 'prew'
    else:
        raise ValueError('Invalid agent!')

    corr_arr = np.zeros((len(xlst), len(ylst))) * np.nan
    for i, xval in enumerate(tqdm.tqdm(xlst)):
        for j, yval in enumerate(ylst):
            print(f"i = {i}, gamma = {xval}, j = {j}, eps = {yval}")
            params[xlabel] = xval
            params[ylabel] = yval
            Ne, Nr = run_rew_error_simulations(params)

            y1 = Ne
            X1 = Nr[:, np.newaxis]
            X1 = sm.add_constant(X1)
            model = sm.OLS(y1, X1)
            res = model.fit()
            # corr_arr[i][j] = res.params[1]
            corr_arr[i][j] = np.corrcoef(Nr, Ne)[0,1]

    return corr_arr






def run_repeated_single_agent(params):
    '''
    params: dictionary of parameters
    '''
    np.random.seed(params['seed'])

    N_iters = params['N_iters']
    verbose = 1 - params['verbose']

    T11lst = np.zeros(N_iters)
    T22lst = np.zeros(N_iters)
    E1lst = np.zeros(N_iters)
    E2lst = np.zeros(N_iters)
    efflist = np.zeros(N_iters)
    PRslopelist = np.zeros(N_iters)
    PLslopelist = np.zeros(N_iters)
    PRoffsetlist = np.zeros(N_iters)
    PLoffsetlist = np.zeros(N_iters)
    LapseL = np.zeros(N_iters)
    LapseR = np.zeros(N_iters)
    ParamsA = np.zeros(N_iters)
    ParamsB = np.zeros(N_iters)
    ParamsC = np.zeros(N_iters)


    for idx in tqdm.tqdm(range(N_iters), disable=verbose):
        # if idx % 10 == 0:
        #     print('Iteration = ', idx)

        agent, world, pR, pL, hmm = run_single_agent(0, 0, params)

        xvals, means, _, _ = simulate_rew_error_correlations(world, agent)
        paramsFit, _ = fit_expfun2([0.5, 4], xvals, np.array(means))

        efflist[idx] = agent.find_efficiency()

        if params['hmm_fit']:
            T11lst[idx] = hmm.transitions.transition_matrix[0][0]
            T22lst[idx] = hmm.transitions.transition_matrix[1][1]
            E1lst[idx] = logistic(hmm.observations.logit_ps)[0]
            E2lst[idx] = logistic(hmm.observations.logit_ps)[1]

        PRslopelist[idx] = pR[0]
        PLslopelist[idx] = pL[0]
        PRoffsetlist[idx] = pR[1]
        PLoffsetlist[idx] = pL[1]
        LapseL[idx] = pL[2]
        LapseR[idx] = pR[2]
        ParamsA[idx] = paramsFit[0]
        ParamsB[idx] = paramsFit[1]
        # ParamsC[idx] = paramsFit[2]

    return efflist, T11lst, T22lst, E1lst, E2lst, PRslopelist, PLslopelist, \
           PRoffsetlist, PLoffsetlist, LapseL, LapseR, ParamsA, ParamsB, ParamsC

def run_multiple_agents(params):
    '''
    params: dictionary of parameters
    '''
    np.random.seed(params['seed'])

    if params['type']== 'inf-based':
        xlst = params['pswitchlst'] #np.linspace(0.01, 0.45, 10)
        ylst = params['prewlst'] #np.linspace(0.55, 0.99, 10)
    elif params['type'] == 'qlearning':
        xlst = params['gammalst']  # np.linspace(0.01, 0.45, 10)
        ylst = params['epslst']  # np.linspace(0.55, 0.99, 10)
    elif params['type'] == 'v-accumulation':
        xlst = params['gammalst']
        ylst = params['betalst']

    T11lst = np.zeros((len(xlst), len(ylst)))
    T22lst = np.zeros((len(xlst), len(ylst)))
    E1lst = np.zeros((len(xlst), len(ylst)))
    E2lst = np.zeros((len(xlst), len(ylst)))
    efflist = np.zeros((len(xlst), len(ylst)))
    PRslopelist = np.zeros((len(xlst), len(ylst)))
    PLslopelist = np.zeros((len(xlst), len(ylst)))
    PRoffsetlist = np.zeros((len(xlst), len(ylst)))
    PLoffsetlist = np.zeros((len(xlst), len(ylst)))
    LapseL = np.zeros((len(xlst), len(ylst)))
    LapseR = np.zeros((len(xlst), len(ylst)))
    ParamsA = np.zeros((len(xlst), len(ylst)))
    ParamsB = np.zeros((len(xlst), len(ylst)))
    ParamsC = np.zeros((len(xlst), len(ylst)))

    coefs_all = []
    perf_train_all = []
    perf_test_all = []

    for idx in range(len(xlst)):
        print('* idx = ', idx)
        for idy in range(len(ylst)):
            print('     idy = ', idy)
            agent, world, pR, pL, hmm = run_single_agent(idx, idy, params)

            coefs, perf_train, perf_test = agent.do_history_logistic_fit(Tmax=params['Tmax'])
            coefs_all.append(coefs)
            perf_train_all.append(perf_train)
            perf_test_all.append(perf_test)

            # Previous fitting code
            xvals, means, _, _ = simulate_rew_error_correlations(world, agent)
            paramsFit, _ = fit_expfun2([0.5, 4], xvals, np.array(means))
            efflist[idx][idy] = agent.find_efficiency()

            if params['hmm_fit']:
                T11lst[idx][idy] = hmm.transitions.transition_matrix[0][0]
                T22lst[idx][idy] = hmm.transitions.transition_matrix[1][1]
                E1lst[idx][idy] = logistic(hmm.observations.logit_ps)[0]
                E2lst[idx][idy] = logistic(hmm.observations.logit_ps)[1]

            PRslopelist[idx][idy] = pR[0]
            PLslopelist[idx][idy] = pL[0]
            PRoffsetlist[idx][idy] = pR[1]
            PLoffsetlist[idx][idy] = pL[1]
            LapseL[idx][idy] = pL[2]
            LapseR[idx][idy] = pR[2]
            ParamsA[idx][idy] = paramsFit[0]
            ParamsB[idx][idy] = paramsFit[1]

    # Reshape the logistic regression results
    coefs_arr = np.array(coefs_all)
    coefs_arr = np.reshape(coefs_arr, [len(xlst), len(ylst), -1])
    perf_train_arr = np.reshape(perf_train_all, [len(xlst), -1])
    perf_test_arr = np.reshape(perf_test_all, [len(xlst), -1])

    return efflist, T11lst, T22lst, E1lst, E2lst, PRslopelist, PLslopelist, \
           PRoffsetlist, PLoffsetlist, LapseL, LapseR, ParamsA, ParamsB, ParamsC, \
            coefs_arr, perf_train_arr, perf_test_arr



def run_single_agent(idx, idy, params):
    '''
    For running a single set of parameters
    '''
    # print(params)
    rlow = params['rlow']
    rhigh = params['rhigh']
    nblocks = params['nblocks']
    ntrials_per_block = params['ntrials_per_block']
    obs_dim = params['obs_dim']
    num_states = params['num_states']
    N_iters = params['N_iters']
    window = params['sigmoid_window']
    # hmm_fit = params['hmm_fit']

    world, _ = src.utils.make_switching_world(rlow, rhigh, nblocks, ntrials_per_block[0], ntrials_per_block[1])

    if params['type'] == 'inf-based':
        agent = EGreedyInferenceBasedAgent(prew=params['prewlst'][idy], pswitch=params['pswitchlst'][idx], eps=params['eps'])
    elif params['type'] == 'qlearning':
        agent = EGreedyQLearningAgent(gamma=params['gammalst'][idx], eps=params['epslst'][idy])
    elif params['type'] == 'v-accumulation':
        agent = ValueAccumulationAgent(gamma=params['gammalst'][idx], beta=params['betalst'][idy])
    else:
        raise ValueError('Agent not recognized, must be inf-based, qlearning, or v-accumulation')

    exp = Experiment(agent, world)
    exp.run()

    # Fit HMM to choice sequence
    if params['hmm_fit']:
        data = np.array(agent.choice_history)[:, None]
        ## testing the constrained transitions class
        hmm = ssm.HMM(num_states, obs_dim, observations="bernoulli")
        _ = hmm.fit(data, method="em", num_iters=N_iters, init_method="kmeans", verbose=0)
    else:
        hmm = None

    # Sigmoidal fit for choice transitions
    # pR, pL, _ = find_LR_transition_fit(world, agent, window=window, type='sigmoid')
    # print('sigmoid:', pR, pL)
    pR, pL, _ = src.utils.find_LR_transition_fit(world, agent, window=window, type='doublesigmoid')
    # print('doublesigmoid', pR, pL)
    return agent, world, pR, pL, hmm

if __name__ == '__main__':
    # A systematic simulation of the whole space
    # params = dict(nblocks=1000, seed=123, pstruct=[5, 40], agenttype='qlearning', world_pr=0.9, world_ps=0.1,
    #               gamma=np.linspace(0.01, 0.3, 5), eps=np.linspace(0.01, 0.3, 5))
    # corr_arr = rew_err_sweep(params)
    params = dict(nblocks=1000, seed=123, pstruct=[5, 40], agenttype='qlearning', world_pr=0.9, world_ps=0.1,
                  gamma=0.1, eps=0.01)

    Ne, Nr = run_rew_error_simulations(params)