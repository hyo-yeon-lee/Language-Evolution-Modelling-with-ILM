from venv import create

import numpy as np
import torch
import torch.nn as nn
from PIL.ImagePalette import random
from numpy.array_api import float32
import autoencoder as at

# Defining Agent class
class Agent:
    def __init__(self, bitN, m2s, s2m):
        self.bitN = bitN
        self.m2s = m2s # meaning to signal (encoder)
        self.s2m = s2m # sigbnal to meaning (decoder)
        # would work like an autoencoder, feeds the output signal from m2s as an input for s2m
        self.m2m = nn.Sequential(m2s, s2m)
        self.m2sTable = [None] * (2 ** bitN)


# create a näive state of agent
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


def clear_agent(agent):
    agent.bitN = None
    agent.m2s = None
    agent.s2m = None
    agent.m2m = None
    agent.m2sTable = None


# helper functions : integer -> binary vectors -> integer
def init_m2sTable(agent):
    for i in range(2 ** agent.bitN):
        meaning = torch.tensor(int2bin(agent.bitN, i), dtype=float32)
        probs = agent.m2s(meaning)
        signal = probs.round().detach().numpy()
        agent.m2sTable[i] = bin2ints(signal) + 1

def int2bin(bitN, value):
    return [int(x) for x in f"{value:0{bitN}b}"]


def bin2ints(bin_vec):
    return int("".join(map(str, bin_vec)), 2) #first convert and join them to a string and change it to int


def gen_random_table(bitN):
    return np.random.permutation(bitN)

# Make table functions
def make_table(agent):
    table = [0] * (2 ** agent.bitN)
    mapping = np.random.permutation(range(agent.bitN)) # shuffling the bit order
    flip= np.random.randint(0, 2, agent.bitN) #flip the bits randomly
    # mapping = np.arange(agent.bitN)
    # flip = np.random.choice([0, 1], size=agent.bitN, p=[0.8, 0.2])  # Flip ~20% of bits
    # np.random.shuffle(mapping[:agent.bitN // 2])  # Shuffle only half of the bits

    for i in range(2**agent.bitN):
        num = int2bin(agent.bitN, i)
        new_num = [num[mapping[j]] ^ flip[j] for j in range(agent.bitN)]
        table[i] = bin2ints(new_num) + 1
    return None

def gen_supervised_data(agent, tutor, batch_size):
    data = [(int2bin(agent.bitN, i), tutor.m2sTable[i]) for i in range(2 ** agent.bitN)]
    np.random.shuffle(data)

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# use when the supervised dataset is different from unsupervised training dataset
def gen_unsupervised_data(agent, batch_size):
    data = [int2bin(agent.bitN, i) for i in range(2 ** agent.bitN)]
    np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# Implement training functions
def train_supervised(agent, tutor, batch_size = 8, epochs = 10):
    train_encoder = np.random.randint(0, 1)
    optimizer = torch.optim.Adam(
        list(agent.m2s.parameters()) + list(agent.s2m.parameters()), lr = 0.01
    )

    # MSE loss will calculate Mean Squared Error between the inputs
    loss_function = nn.MSELoss()

    print('====Training start====')
    for epoch in range(epochs):
        train_loss = 0
        for batch in gen_supervised_data(agent, tutor, batch_size):
            # prepare input data
            # data = data.cuda() # use when the input is no longer vector?
            # inputs = torch.reshape(data, (-1, 784))  # -1 can be any value. So when reshape, it will satisfy 784 first

            meanings, signals = zip(*batch)
            meanings = torch.tensor(meanings, dtype=torch.float32)
            signals = torch.tensor(signals, dtype=torch.float32)

            # set gradient to zero train encoder(m2s)
            optimizer.zero_grad()
            pred_m2s = agent.m2s(meanings)
            loss_m2s = loss_function(pred_m2s, signals)

            #train decoder (s2m)
            pred_s2m = agent.s2m(signals)
            loss_s2m = loss_function(pred_s2m, meanings)

            # calculating loss
            loss = loss_m2s + loss_s2m

            # calculate gradient of each parameter
            loss.backward()
            train_loss += loss.item()

            # update the weight based on the gradient calculated
            optimizer.step()
        print(f"Epoch {epoch + 1}, Total Loss: {train_loss:.4f}, m2s Loss: {loss_m2s}, s2m Loss: {loss_s2m}")


def train_unsupervised(agent):
    return None


# simulate iterated learning
def iterated_learning(bitN = 8, layerN = 1):
    tutor = create_agent(bitN, layerN)

    return None

# Evaluation
def calculate_entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def stability(tutor, pupil):
    return sum(1 for t, p in zip(tutor, pupil) if t == p) / len(tutor)

def expressivity(agent):
    return None


def main():
    return None


if __name__ == "__main__":
    #set params here
    main()