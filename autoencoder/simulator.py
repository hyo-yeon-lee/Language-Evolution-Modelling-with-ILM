import numpy as np
import torch
import torch.nn as nn
from numpy.array_api import float32

import autoencoder as at

# Defining Agent class
class Agent:
    def __init__(self, bitN, m2s, s2m):
        self.bitN = bitN
        self.m2s = m2s # meaning to signal (encoder)
        self.s2m = s2m # signal to meaning (decoder)
        # would work like an autoencoder, feeds the output signal from m2s as an input for s2m
        self.m2m = nn.Sequential(m2s, s2m)
        self.m2sTable = [None] * (2 ** bitN)


def create_agent(bitN, layerN):
    m2s = nn.Sequential(
        nn.Linear(bitN, layerN),
        nn.Sigmoid(),
        nn.Linear(layerN, bitN),
        nn.Sigmoid()
    )
    s2m = nn.Sequential(
        nn.Linear(bitN, layerN),
        nn.Sigmoid(),
        nn.Linear(layerN, bitN),
        nn.Sigmoid()
    )

    return Agent(bitN, m2s, s2m)


# def clear_agent(agent):
#     agent.bitN = None
#     agent.m2s = None
#     agent.s2m = None
#     agent.m2m = None


# helper functions : integer -> binary vectors -> integer
def init_m2sTable(agent):
    for i in range(2 ** agent.bitN):
        meaning = torch.tensor(int2bin(agent.bitN, i), dtype=float32)
        probs = agent.m2s(meaning)
        signal = probs.round().detach().numpy()
        agent.m2sTable[i] = bin2ints(signal) + 1

def int2bin(bitN, value):
    return torch.tensor([(value >> i) & 1 for i in range(bitN - 1, -1, -1)], dtype=torch.float32)

def bin2ints(value):
    return None


# Make table functions
def make_table(agent):
    return None

# Implement training functions
def train_supervised(agent):
    return None

def train_unsupervised(agent):
    return None

# Transition pupil to tutor
def transit_p2t(pupil, tutor):
    return None

# simulate iterated learning
def iterated_learning():
    return None

# Evaluation
def calculate_entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def stability():
    return None

def expressivity(agent):
    return None


def main():
    return None


if __name__ == "__main__":
    #set params here
    expressivity()