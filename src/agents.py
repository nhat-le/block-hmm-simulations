import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import src.utils

class Agent():
    '''
    A general agent
    '''

    def __init__(self):
        self.outcome_history = []
        self.choice_history = []
        self.Rewards1side_history = []
        self.Rewards0side_history = []

    def find_efficiency(self):
        return sum(self.outcome_history) / len(self.outcome_history)

    def find_prob(self):
        '''
        Find the instantaneous probability of the agent for all trials
        '''
        p0 = np.array(self.Rewards0side_history)
        p1 = np.array(self.Rewards1side_history)
        return p1 / (p0 + p1)

    def get_running_choice_fraction(self, window):
        '''
        Get the running mean of choice fraction
        window: int, window for averaging
        '''
        choice_arr = np.array(self.choice_history)
        kernel = np.ones(window) / window
        return np.convolve(choice_arr, kernel, 'same')

    def get_running_reward_fraction(self, window):
        '''
        Get the running mean of reward fraction
        window: int, window for averaging
        '''
        rewards0side = np.array(self.Rewards0side_history)
        rewards1side = np.array(self.Rewards1side_history)
        kernel = np.ones(window) / window

        running_0sidefrac = np.convolve(rewards0side, kernel, 'same')
        running_1sidefrac = np.convolve(rewards1side, kernel, 'same')

        return running_1sidefrac / (running_0sidefrac + running_1sidefrac)


    def do_history_logistic_fit(self, Tmax):
        choice_t = self.choice_history[10:0]
        choices_t = [np.array(self.choice_history[(Tmax - i):(-i - 1)]) * 2 - 1 for i in range(Tmax)]
        rewards_t = [np.array(self.outcome_history[(Tmax - i):(-i - 1)]) for i in range(Tmax)]

        RewC_t = [choices_t[i] * rewards_t[i] for i in range(len(choices_t))]
        #UnrC_t = [choices_t[i] * (1 - rewards_t[i]) for i in range(len(choices_t))]

        X = np.vstack(choices_t[1:] + RewC_t[1:] + rewards_t[1:]).T
        y = choices_t[0]

        # Logistic regression
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)
        model = LogisticRegression(random_state=0, C=1).fit(Xtrain, ytrain)

        perf_train = np.sum(model.predict(Xtrain) == ytrain) / len(ytrain)
        perf_test = np.sum(model.predict(Xtest) == ytest) / len(ytest)

        return model.coef_.flatten(), perf_train, perf_test


class MatchingAgent(Agent):
    '''
    Simulate an agent that matches perfectly (perfect integration of past rewards)
    '''

    def __init__(self, eps):
        '''
        eps: rate is limited to the range [eps, 1-eps]
        '''
        assert (0 <= eps <= 0.5)
        self.Rewards1side = 1  # Start with 1 so that the agent does not get 'stuck'
        self.Rewards0side = 1
        self.eps = eps
        self.choice_history = []
        self.Rewards1side_history = []
        self.Rewards0side_history = []
        self.outcome_history = []

    def outcome_received(self, outcome):
        if outcome == 1:
            if self.choice_history[-1] == 1:
                self.Rewards1side += 1
            else:
                self.Rewards0side += 1

        self.outcome_history.append(outcome)
        self.Rewards1side_history.append(self.Rewards1side)
        self.Rewards0side_history.append(self.Rewards0side)

    def make_choice(self):
        '''
        Make a choice, probabilistically sample from past reward ratios
        '''

        p = self.Rewards1side / (self.Rewards1side + self.Rewards0side)
        p = min(p, 1 - self.eps)
        p = max(p, self.eps)
        choice = np.random.rand() < p
        self.choice_history.append(choice)
        return choice


class EGreedyQLearningAgent(Agent):
    '''
    Simulate an agent that uses Q-learning, and chooses action probabilistically
    based on the ratio of the q-values
    '''

    def __init__(self, gamma, eps=0):
        '''
        gamma: learning rate for q-value updates
        eps: used for epsilon-greedy strategy, eps = 0 means a greedy agent
        '''
        self.q0 = 0.5
        self.q1 = 0.5
        self.gamma = gamma
        self.eps = eps
        self.choice_history = []
        self.q0_history = []
        self.q1_history = []
        self.outcome_history = []
        self.Rewards0side_history = []
        self.Rewards1side_history = []

    def outcome_received(self, outcome):
        if self.choice_history[-1] == 1:
            self.q1 = self.q1 + self.gamma * (outcome - self.q1)
        else:
            self.q0 = self.q0 + self.gamma * (outcome - self.q0)

        self.outcome_history.append(outcome)
        self.q0_history.append(self.q0)
        self.q1_history.append(self.q1)

        # Update reward history on each side
        if outcome == 1:
            if self.choice_history[-1] == 1:
                self.Rewards1side_history.append(1)
                self.Rewards0side_history.append(0)
            else:
                self.Rewards1side_history.append(0)
                self.Rewards0side_history.append(1)

    def make_choice(self):
        '''
        Make a choice using the epsilon-greedy strategy
        '''
        # Flip a coin to decide if explore or exploit
        explore = np.random.rand() < self.eps
        if explore:  # choose actions randomly
            choice = int(np.random.rand() < 0.5)
        else:
            if self.q1 > self.q0:
                choice = 1
            elif self.q1 < self.q0:
                choice = 0
            else:
                choice = int(np.random.rand() < 0.5)

        self.choice_history.append(choice)
        return choice

    def find_prob(self):
        '''
        Find the instantaneous probability of the agent for all trials
        '''
        p0 = np.array(self.q0_history)
        p1 = np.array(self.q1_history)
        return p1 / (p0 + p1)







class ValueAccumulationAgent(Agent):
    '''
    Simulate an agent that uses value accumulation as described in Mainen's paper
    '''

    def __init__(self, gamma, beta=10):
        '''
        gamma: learning rate for value updates
        beta: the sharpness of the sigmoidal curve used to generate the behavior
        '''
        self.V = 0
        self.gamma = gamma
        self.beta = beta
        self.choice_history = []
        self.v_history = []
        self.outcome_history = []
        self.Rewards0side_history = []
        self.Rewards1side_history = []

    def outcome_received(self, outcome):
        # Note that previous choice is 0 or 1 -> need to convert to 1 or -1
        prevchoice = self.choice_history[-1] * 2 - 1 #[0,1] -> [-1, 1]
        self.V = (1 - self.gamma) * self.V + self.gamma * outcome * prevchoice

        self.outcome_history.append(outcome)
        self.v_history.append(self.V)

        # Update reward history on each side
        if outcome == 1:
            if self.choice_history[-1] == 1:
                self.Rewards1side_history.append(1)
                self.Rewards0side_history.append(0)
            else:
                self.Rewards1side_history.append(0)
                self.Rewards0side_history.append(1)

    def make_choice(self):
        '''
        Make a choice using a probabilistic strategy based on current V
        '''
        prob = src.utils.logistic(self.V * self.beta)

        # Flip a coin to choose behavior based on prob
        choice = int(np.random.rand() < prob)
        self.choice_history.append(choice)
        return choice

    def find_prob(self):
        '''
        Find the instantaneous probability of the agent for all trials
        '''
        return self.v_history




# A Matching agent object
class ConstantProbAgent(Agent):
    '''
    Simulate an agent that decides with a fixed probability
    '''

    def __init__(self, prob):
        self.choice_history = []
        self.prob = prob
        self.outcome_history = []

    #         self.Rewards1side_history = []
    #         self.Rewards0side_history = []

    def outcome_received(self, outcome):
        self.outcome_history.append(outcome)

    def make_choice(self):
        '''
        Make a choice, probabilistically sample from past reward ratios
        '''
        choice = np.random.rand() < self.prob
        self.choice_history.append(choice)
        return choice

    # A Matching agent object


class PiecewiseConstantProbAgent(Agent):
    '''
    Simulate an agent that decides with a fixed probability in blocks
    'Optimal' agent when it knows the probability of each block in a world
    '''

    def __init__(self, rates, ntrials):
        self.choice_history = []
        self.rates = rates
        self.ntrials = ntrials
        self.outcome_history = []
        self.curr_rate = rates[0]
        self.curr_block = 0

    def outcome_received(self, outcome):
        self.outcome_history.append(outcome)

    def make_choice(self):
        '''
        Make a choice, probabilistically sample from past reward ratios
        '''
        choice = np.random.rand() < self.curr_rate

        if len(self.outcome_history) > sum(self.ntrials[:self.curr_block + 1]):
            self.curr_block += 1
            self.curr_rate = self.rates[self.curr_block]

        self.choice_history.append(choice)
        return choice


class EGreedyInferenceBasedAgent(Agent):
    def __init__(self, prew, pswitch, eps=0):
        '''
        An inference-based agent
        :param prew: probability of reward of high-state
        :param pswitch: probability of switch
        :param eps: value used for epsilon-greedy strategy
        '''
        self.prew = prew
        self.pswitch = pswitch
        self.choice_history = []
        self.p0_history = []
        self.p1_history = []
        self.eps = eps
        self.outcome_history = []
        self.Rewards0side_history = []
        self.Rewards1side_history = []
        self.type = type
        self.p0 = 0.5
        self.p1 = 0.5

    def outcome_received(self, outcome):
        self.p0, self.p1 = self.update_prob(self.choice_history[-1], outcome)

        self.outcome_history.append(outcome)
        self.p0_history.append(self.p0)
        self.p1_history.append(self.p1)

        # Update reward history on each side
        if outcome == 1:
            if self.choice_history[-1] == 1:
                self.Rewards1side_history.append(1)
                self.Rewards0side_history.append(0)
            else:
                self.Rewards0side_history.append(1)
                self.Rewards1side_history.append(0)

    def update_prob(self, choice, outcome):
        '''
        Returns the updated probability (p0, p1)
        :param choice: previous choice, 0 or 1
        :param outcome: previous outcome, 0 or 1
        :return: updated probability (p0, p1)
        '''
        prew = self.prew
        pswitch = self.pswitch

        if choice == 1 and outcome == 1:  # chose right, rew
            LLHrtGiven0 = 1 - prew
            LLHrtGiven1 = prew
        elif choice == 0 and outcome == 1:  # chose left, rew
            LLHrtGiven0 = prew
            LLHrtGiven1 = 1 - prew
        elif choice == 1 and outcome == 0:  # chose right, no rew
            LLHrtGiven0 = prew
            LLHrtGiven1 = 1 - prew
        else:  # chose left, no rew
            LLHrtGiven0 = 1 - prew
            LLHrtGiven1 = prew

        p1prev = self.p1
        p0prev = self.p0
        p0new = (1 - pswitch) * LLHrtGiven0 * p0prev + \
                pswitch * LLHrtGiven1 * p1prev
        p1new = pswitch * LLHrtGiven0 * p0prev + \
                (1 - pswitch) * LLHrtGiven1 * p1prev

        p0 = p0new / (p0new + p1new)
        p1 = p1new / (p0new + p1new)

        return p0, p1

    def make_choice(self):
        explore = np.random.rand() < self.eps
        if explore:
            choice = int(np.random.rand() < 0.5)
        else:
            # Optimal agent picks the action with higher prob
            if self.p1 > self.p0:
                choice = 1
            elif self.p1 < self.p0:
                choice = 0
            else:
                choice = int(np.random.rand() > 0.5)

            # Old code for random agent
            # choice = np.random.rand() > self.p0 / (self.p0 + self.p1)
        self.choice_history.append(choice)
        return choice


class LocalMatchingAgent(Agent):
    '''
    Simulate an agent that matches with a leaky integrator
    '''

    def __init__(self, tau, eps):
        """
        tau: time constant of matching agent (integration kernel)
        eps: rate is limited to the range [eps, 1-eps]
        """
        assert (0 <= eps <= 0.5)
        self.tau = tau
        self.eps = eps
        self.Rewards1side = 1  # Start with 1 so that the agent does not get 'stuck'
        self.Rewards0side = 1
        self.choice_history = []
        self.Rewards1side_history = []
        self.Rewards0side_history = []
        self.outcome_history = []

    def outcome_received(self, outcome):
        if self.tau == 0:
            factor = 1
        else:
            factor = np.power(0.5, 1 / self.tau)
        self.Rewards1side = self.Rewards1side * factor
        self.Rewards0side = self.Rewards0side * factor
        if outcome == 1:
            if self.choice_history[-1] == 1:
                self.Rewards1side += 1
            else:
                self.Rewards0side += 1

        self.outcome_history.append(outcome)
        self.Rewards1side_history.append(self.Rewards1side)
        self.Rewards0side_history.append(self.Rewards0side)

    def make_choice(self):
        '''
        Make a choice, probabilistically sample from past reward ratios
        '''

        p = self.Rewards1side / (self.Rewards1side + self.Rewards0side)
        p = min(p, 1 - self.eps)
        p = max(p, self.eps)
        choice = np.random.rand() < p
        self.choice_history.append(choice)
        return choice
