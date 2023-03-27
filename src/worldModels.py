import numpy as np
import matplotlib.pyplot as plt


class World():
    '''
    A general class for worlds
    '''
    def __init__(self):
        self.history = []
        self.rate_history = []

    def find_prob(self):
        '''
        Returns array of rates for trial-1 side
        '''
        rateArr = np.array(self.rate_history)
        return rateArr[:,1]

    def get_rate_history(self):
        return np.array(self.rate_history)


# A world object
class RandomWorld(World):
    '''
    A probablistic world that alternates between blocks of constant probability
    '''
    def __init__(self, rates, ntrials):
        '''
        rates: a list, rates of the world
        ntrials: a list, number of trials in each block
        '''
        self.rates = rates
        self.ntrials = ntrials
        self.curr_block = 0
        self.history = []
        self.rate_history = []
        self.curr_rate = self.rates[0]
        self.curr_side = np.random.rand() < self.curr_rate
        #print('curr side=  ', int(self.curr_side))
        
    def update(self, agent_choice):
        '''
        Update the world based on agent choice
        '''
        self.history.append(self.curr_side)
        self.rate_history.append(self.curr_rate)
        
        if agent_choice == self.curr_side:
            reward = 1
            #print('len history = ', len(self.history), 'next switch at', sum(self.ntrials[:self.curr_block + 1]))
            # See if the world should switch blocks
            if len(self.history) > sum(self.ntrials[:self.curr_block + 1]):
#                 print('world switching!')
                self.curr_block += 1
                self.curr_rate = self.rates[self.curr_block]
                        
            # Sample the next side
            self.curr_side = np.random.rand() < self.curr_rate
        
        else:
            # Incorrect, keep the same side
            reward = 0
                    
        return reward
    

# A persistent world object
class PersistentWorld(World):
    '''
    A probablistic world that alternates between blocks of constant probability
    At each trial, rewards are chosen based on the underlying rate at each site
    If a site becomes active, it stays active until the agent selects it
    '''
    def __init__(self, rates, ntrials):
        '''
        rates: a nblocks x 2 array, representing rates for 0- and 1-sides
        ntrials: a list, number of trials in each block
        '''
        self.rates = rates
        self.ntrialblocks = [0]
        # self.ntrialblockseen = [0]
        self.nblockmax = len(ntrials)
        self.ntrials = ntrials
        self.curr_block = 0
        self.side_history = []
        self.rate_history = []
        
        self.curr_rates = np.array(self.rates[0,:])        
        self.active_sites = np.random.rand(2) < rates[0,:]
        
#         print('curr active sites:', self.active_sites)
        
        #print('curr side=  ', int(self.curr_side))


    def reset(self):
        #TODO
        return None

    def update(self, agent_choice):
        '''
        Update the world based on agent choice
        '''
        agent_choice = int(agent_choice)
        self.rate_history.append(self.curr_rates.copy())
        self.side_history.append(self.active_sites.copy())
        
        # Is there reward at current choice side?
        reward = self.active_sites[agent_choice]
        self.ntrialblocks[-1] = self.ntrialblocks[-1] + 1
#         print('choice = ', agent_choice, 'reward = ', reward)
#         print('n trials so far =', len(self.side_history))
        
        
        # Are we switching blocks?
        if self.curr_block < len(self.ntrials) and len(self.rate_history) >= sum(self.ntrials[:self.curr_block + 1]):
            self.curr_block += 1

            # If we are at the end, don't need to update curr_rates
            if self.curr_block < len(self.ntrials):
                self.curr_rates = self.rates[self.curr_block,:]
            self.ntrialblocks.append(0)
#             print('world switching! curr rates = ', self.curr_rates, 'trials so far =', len(self.side_history))
        
        
        # Update active_sites
        if self.active_sites[0] == 0 or agent_choice == 0:
#             print('updated site 0')
            self.active_sites[0] = np.random.rand() < self.curr_rates[0]

        if self.active_sites[1] == 0 or agent_choice == 1:
#             print('updated site 1', self.curr_rates[1])
            self.active_sites[1] = np.random.rand() < self.curr_rates[1]

#         print('current active sites: ', self.active_sites)
            
                    
        return reward


class PersistentWorldWithCheck(World):
    '''
    A probablistic world that alternates between blocks of constant probability,
    only switching after performance crosses a threshold
    '''

    def __init__(self, rates, ntrials, threshold):
        '''
        rates: a nblocks x 2 array, representing rates for 0- and 1-sides
        ntrials: a list, number of trials in each block
        '''
        self.rates = rates
        self.threshold = threshold
        self.currperf = -1 # Current performance in the block

        self.currCorrect = 0
        self.currIncorr = 0

        self.ntrials = ntrials
        self.curr_block = 0
        self.side_history = []
        self.rate_history = []

        self.curr_rates = np.array(self.rates[0, :])
        self.active_sites = np.random.rand(2) < rates[0, :]

    def update(self, agent_choice):
        '''
        Update the world based on agent choice
        '''
        agent_choice = int(agent_choice)
        self.rate_history.append(self.curr_rates.copy())
        self.side_history.append(self.active_sites.copy())

        # Is there reward at current choice side?
        reward = self.active_sites[agent_choice]

        if reward > 0:
            self.currCorrect += 1
        else:
            self.currIncorr += 1

        self.currperf = self.currCorrect / (self.currCorrect + self.currIncorr)
        #print(self.currperf)


        # Are we switching blocks?
        if self.currCorrect + self.currIncorr > self.ntrials[self.curr_block] and \
            self.currperf > self.threshold:
            #print('Block switching!')
            self.curr_block += 1
            self.curr_rates = self.rates[self.curr_block, :]
            self.currCorrect = 0
            self.currIncorr = 0
            self.currperf = -1

        # Update active_sites
        if self.active_sites[0] == 0 or agent_choice == 0:
            #             print('updated site 0')
            self.active_sites[0] = np.random.rand() < self.curr_rates[0]

        if self.active_sites[1] == 0 or agent_choice == 1:
            #             print('updated site 1', self.curr_rates[1])
            self.active_sites[1] = np.random.rand() < self.curr_rates[1]

        #         print('current active sites: ', self.active_sites)

        return reward


# A foraging world object
class ForagingWorld(World):
    '''
    A probablistic world that alternates between blocks of constant probability
    In each block, there is a designated 'active site', which becomes active
    with some probability prew. The non-active site is always non-active
    Each trial at an active site causes a switch with probability psw
    '''

    def __init__(self, prew, psw, pstruct, nblockmax):
        '''
        rates: a nblocks x 2 array, representing rates for 0- and 1-sides
        ntrials: a list, number of trials in each block
        maxtrials: number of trials to simulate
        pstruct: a tuple of (N1, N2) indicating the blocks for structure of transitions
        from [0, N1), psw = 0, from [N1, N2), psw = psw, from [N2, inf), psw = 1
        '''
        self.prew = prew
        self.psw = psw
        self.curr_block = 0
        self.side_history = []
        self.pstruct = pstruct
        self.nblockmax = nblockmax
        self.ntrialblocks = [0] #an array for keeping track of how many trials are there in the blocks
        # note that ntrialblocks[-1] indicates the current count of the block
        self.currside_history = []
        # self.ntrials = [ntrials]
        self.rate_history = []

        self.curr_side = int(np.random.rand() < 0.5)
        # self.active_sites = np.array([False, False])
        # self.active_sites[self.curr_side] = np.random.rand() < prew

    #         print('curr active sites:', self.active_sites)

    # print('curr side=  ', int(self.curr_side))

    def reset(self):
        #TODO
        return None

    def update(self, agent_choice):
        '''
        Update the world based on agent choice
        '''
        self.ntrialblocks[-1] += 1

        # if self.ntrialblocks[-1] > 15:
        #     print('here')

        # What is the psw at the moment?
        if self.ntrialblocks[-1] < self.pstruct[0]:
            psw_curr = 0
        elif self.ntrialblocks[-1] < self.pstruct[1]:
            psw_curr = self.psw
        else:
            psw_curr = 1

        agent_choice = int(agent_choice)
        # self.currside_history.append(self.curr_side.copy())
        self.side_history.append(self.curr_side)
        # print('Current site = ', self.curr_side)
        ratearr = [False, False]
        ratearr[self.curr_side] = True
        self.rate_history.append(ratearr)

        # Is there reward at current choice side?
        if agent_choice == self.curr_side:
            reward = int(np.random.rand() < self.prew)
        else:
            reward = 0
        #         print('choice = ', agent_choice, 'reward = ', reward)
        #         print('n trials so far =', len(self.side_history))

        # Are we switching blocks?
        if agent_choice == self.curr_side and np.random.rand() < psw_curr:
            self.curr_block += 1
            self.curr_side = 1 - self.curr_side
            self.ntrialblocks.append(0)
            # print('Block has switched!, current active site is', self.curr_side)
            # self.curr_rates = self.rates[self.curr_block, :]
        #             print('world switching! curr rates = ', self.curr_rates, 'trials so far =', len(self.side_history))

        # Update active sites
        # print('Choice = ', agent_choice, 'Reward =', reward)


        return reward


    
class Experiment():
    '''
    An experiment consists of an agent in a world
    '''
    def __init__(self, agent, world):
        self.agent = agent
        self.world = world
        
        
    def run(self):
        '''
        Run the proposed experiment
        '''
        choices = []
        rewards = []

        # counter = 0
        # print()
        while len(self.world.ntrialblocks) <= self.world.nblockmax:
            # print(counter)
            # counter += 1
        # for i in range(sum(self.world.ntrials)):
            choice = self.agent.make_choice()
            #print('choice = ', int(choice))
            choices.append(choice)
            reward = self.world.update(choice)
            #print('reward = ', int(reward))
            self.agent.outcome_received(reward)
            rewards.append(reward)
            
        return choices, rewards


    def visualize(self):
        '''
        Visualize the performance of the agent in the experiment
        '''
        agent = self.agent
        world = self.world
        plt.figure()
        # plt.plot(agent.find_prob())
        plt.plot(agent.q1_history, '.')
        plt.plot(agent.q0_history, '.')
        plt.plot(np.array(agent.choice_history) * 2 - 1, '.')

        blockswitches = np.cumsum(world.ntrialblocks)
        for i in range(len(world.ntrialblocks) - 1):
            if i % 2:
                plt.fill_between([blockswitches[i], blockswitches[i + 1]], [-1, -1], [1, 1], color='r', alpha=0.2)
            else:
                plt.fill_between([blockswitches[i], blockswitches[i + 1]], [-1, -1], [1, 1], color='b', alpha=0.2)

        plt.xlim([0, 400])



