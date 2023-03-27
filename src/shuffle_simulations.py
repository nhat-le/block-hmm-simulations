import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import scipy.io
import src.utils

def generate_blocks_with_offset(Nblocks, blocklen, offset):
    '''
    Generate a session with Nblocks blocks, each block of length
    blocklen trials, and all of them has an offset given by offset
    '''
    blocks = np.zeros((Nblocks, blocklen))
    blocks[:,:offset] = 1
    return blocks

def interleave(block1, block2, Ndiv):
    '''
    Interleave behavior from two modes
    block1 and block2 must have the same number of rows
    Ndiv: number of subblocks to divide the behavior
    '''
    assert(block1.shape[0] == block2.shape[0])
    Tsubblock = int(block1.shape[0] / Ndiv)

    block = []
    for i in range(Ndiv):
        block.append(block1[i * Tsubblock : (i + 1) * Tsubblock])
        block.append(block2[i * Tsubblock : (i + 1) * Tsubblock])

    return np.vstack(block)




def generate_blocks_with_sigmoid(Nblocks, blocklen, offset, slope, lapse):
    '''
    Generate a session with Nblocks blocks, each block of length
    blocklen trials, governed by transition function with parameters
    (offset, slope, lapse)
    '''
    x = np.arange(blocklen)
    transfunc = sigmoid(x, offset, slope, lapse)
    blocks = np.random.rand(Nblocks, blocklen) > transfunc
    return blocks, transfunc



def sigmoid(x, offset, slope, lapse):
    return (1 - 2*lapse) / (1 + np.exp(-(x - offset) * slope)) + lapse


def resample_block(block, Ntrials):
    '''
    Resample Ntrials times while maintaining the bigram statistics
    '''
    choicelst = [resample_helper(block) for i in range(Ntrials)]
    return np.array(choicelst)


def resample_helper(block):
    '''
    Resample block a single time maintaining the bigram statistics
    '''
    N, T = block.shape
    choices = []
    for i in range(T):
        if i == 0:
            trials = block[:, i]


        else:
            prevtrials = block[:, i - 1]
            trials = block[prevtrials == sample, i]

        sample = np.random.choice(trials)
        choices.append(sample)

    return choices


def unfold_block(block):
    '''
    Given a block, unfold into three flattened arrays:
    choices, outcomes, choices x outcomes

    '''
    outcomes = (1 - np.reshape(block, (-1, 1))) * 2 - 1
    block = block * 2 - 1
    block[::2, :] = block[::2, :] * -1
    choices = np.reshape(block, (-1, 1))

    return outcomes, choices


def make_Xy(block, N=1):
    '''
    block: raw data block, of shape Nblocks x trials
    N: number of trials back in the past to extract for the regression
    Returns:
    Xmat: Xmatrix of shape (Nblocks x trials - 3N) x (3N),
    arranged as blocks of [choice history; outcome history; choice x outcome history]
    in each block there are N columns, corresponding to (t-N, ..., t-2, t-1)
    y: y vector, the choice

    '''
    outcomes, choices = unfold_block(block)
    outcomesXchoices = outcomes * choices

    y = choices[N:].ravel()

    choicehistory = []
    outcomehistory = []
    choiceoutcomehistory = []

    for i in range(N):
        X1 = choices[i:-N + i]
        X2 = outcomes[i:-N + i]
        X3 = outcomesXchoices[i:-N + i]
        choicehistory.append(X1)
        outcomehistory.append(X2)
        choiceoutcomehistory.append(X3)

    #     print(choicehistory[0].shape, outcomehistory[0].shape, choiceoutcomehistory[0].shape)
    #     print(len(choicehistory), len(outcomehistory), len(choiceoutcomehistory))
    #     print(len(choicehistory + outcomehistory + choiceoutcomehistory))
    Xmat = np.hstack(choicehistory + outcomehistory + choiceoutcomehistory)

    return choicehistory, outcomehistory, choiceoutcomehistory, Xmat, y





