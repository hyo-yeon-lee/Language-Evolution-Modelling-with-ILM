import numpy as np
import torch
import torch.nn as nn
from PIL.ImagePalette import random
from matplotlib import pyplot as plt
from numpy.array_api import float32
# from scipy.special import dtype
from scipy.stats import spearmanr
from sklearn.utils import shuffle


class Agent:
    def __init__(self, bitN, m2s, s2m):
        self.bitN = bitN
        self.m2s = m2s # meaning to signal (encoder)
        self.s2m = s2m # sigbnal to meaning (decoder)
        # would work like an autoencoder, feeds the output signal from m2s as an input for s2m
        self.m2m = nn.Sequential(m2s, s2m)


# create a näive state of agent
def create_agent(bitN, nodeN):
    m2s = nn.Sequential(
        nn.Linear(bitN, nodeN),
        nn.Sigmoid(),
        nn.Linear(nodeN, bitN),
        nn.Sigmoid()
    )
    s2m = nn.Sequential(
        nn.Linear(bitN, nodeN),
        nn.Sigmoid(),
        nn.Linear(nodeN, bitN),
        nn.Sigmoid()
    )

    return Agent(bitN, m2s, s2m)


# helper functions : integer -> binary vectors -> integer
def int2bin(bitN, value):
    return [int(x) for x in f"{value:0{bitN}b}"]


def bin2ints(bin_vec):
    return int("".join(map(str, bin_vec)), 2) #first convert and join them to a string and change it to int


# def gen_random_table(bitN):
#     return np.random.permutation(bitN)


# Creating supervised dataset from tutor's encoder
def gen_supervised_data(agent, tutor, batch_size):
    m2sdata = []

    for _ in range(batch_size):
        meaning = torch.randint(0, 2, (agent.bitN,), dtype=torch.float32)
        signal = tutor.m2s(meaning.unsqueeze(0)).detach().round().squeeze(0)
        m2sdata.append((meaning.numpy(), signal.numpy()))

    s2mdata = m2sdata.copy()
    shuffle(s2mdata)

    return m2sdata, s2mdata


# Creating unsuepervised training data...but do we need it? yes we do
def gen_unsupervised_data(agent, batch_size):
    data = [int2bin(agent.bitN, i) for i in range(2 ** agent.bitN)]
    np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# Combined supervised and unsupervised training
def train_combined(agent, tutor, batch_size=1, epochs=10, autoencoder_iters=20):
    optimiser = torch.optim.SGD(
        list(agent.m2s.parameters()) + list(agent.s2m.parameters()), lr=0.01
    )
    loss_function = nn.MSELoss()

    m2sdata, s2mdata = gen_supervised_data(agent, tutor, batch_size)

    for epoch in range(epochs):
        total_loss = 0

        # Supervised Training : get one data from the batch and train them
        m2s_meaning, m2s_signal = m2sdata[epoch % batch_size]
        s2m_meaning, s2m_signal = s2mdata[epoch % batch_size]

        m2s_meaning = torch.tensor(m2s_meaning, dtype=torch.float32).unsqueeze(0)  # Shape (1, bitN)
        m2s_signal = torch.tensor(m2s_signal, dtype=torch.float32).unsqueeze(0)

        s2m_meaning = torch.tensor(s2m_meaning, dtype=torch.float32).unsqueeze(0)  # Shape (1, bitN)
        s2m_signal = torch.tensor(s2m_signal, dtype=torch.float32).unsqueeze(0)

        optimiser.zero_grad()

        # Train encoder/decoder (m2s/s2m)
        pred_m2s = agent.m2s(m2s_meaning)
        loss_m2s = loss_function(pred_m2s, m2s_signal)

        pred_s2m = agent.s2m(s2m_signal)
        loss_s2m = loss_function(pred_s2m, s2m_meaning)

        loss = loss_m2s + loss_s2m
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

        # Unsupervised Training: 20 iterations of self-reconstruction
        for _ in range(autoencoder_iters):
            meanings_u = torch.randint(0, 2, (batch_size, agent.bitN), dtype=torch.float32)

            optimiser.zero_grad()
            pred_m2m = agent.m2m(meanings_u)
            loss_auto = loss_function(pred_m2m, meanings_u)
            loss_auto.backward()
            optimiser.step()

            total_loss += loss_auto.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# Iterated Learning Process
def iterated_learning(generations=20, bitN=8, nodeN=30, batch_size=1, epochs=10):
    print("\n==== Initializing first tutor (random, untrained network) ====")
    tutor = create_agent(bitN, nodeN)  # No predefined table

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []

    for gen in range(1, generations + 1):
        print(f"\n==== Generation {gen} ====")
        pupil = create_agent(bitN, nodeN)

        # Train using the defined training function
        train_combined(pupil, tutor, batch_size, epochs)

        stability_scores.append(stability(tutor, pupil))
        expressivity_scores.append(expressivity(pupil))
        compositionality_scores.append(compositionality(pupil))

        # Pupil becomes the next tutor
        tutor = pupil

    return stability_scores, expressivity_scores, compositionality_scores



def plot_results(stability_scores, expressivity_scores, compositionality_scores, generations):
    plt.figure(figsize=(15, 5))

    gens = np.arange(1, generations + 1)

    # Stability Plot
    plt.subplot(1, 3, 1)
    plt.plot(gens, stability_scores, color="blue", linewidth=3)
    plt.xlabel("Generations")
    plt.ylabel("s")
    plt.title("A: Stability Over Generations", fontsize=16, loc="left")

    # Expressivity Plot
    plt.subplot(1, 3, 2)
    plt.plot(gens, expressivity_scores, color="orange", linewidth=3)
    plt.xlabel("Generations")
    plt.ylabel("x")
    plt.title("B: Expressivity Over Generations", fontsize=16, loc="left")

    # Compositionality Plot
    plt.subplot(1, 3, 3)
    plt.plot(gens, compositionality_scores, color="green", linewidth=3)
    plt.xlabel("Generations")
    plt.ylabel("c")
    plt.title("C: Compositionality Over Generations", fontsize=16, loc="left")

    plt.suptitle(rf"n = {len(stability_scores)}", fontsize=16, x=0.5, y=1.05)
    plt.tight_layout()
    plt.show()


# Evaluation
def calculate_entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def stability(tutor, pupil, sample_size=1000):
    tutor.m2s.eval()
    pupil.m2s.eval()

    with torch.no_grad():
        matches = 0
        for _ in range(sample_size):
            meaning = torch.randint(0, 2, (tutor.bitN,), dtype=torch.float32).unsqueeze(0)
            tutor_signal = tuple(tutor.m2s(meaning).round().squeeze(0).numpy().astype(int))
            pupil_signal = tuple(pupil.m2s(meaning).round().squeeze(0).numpy().astype(int))

            if tutor_signal == pupil_signal:
                matches += 1

    stability_score = matches / sample_size
    print(f"GETTING STABILITY: {stability_score:.4f}")
    return stability_score


def expressivity(agent):
    agent.m2s.eval()
    with torch.no_grad():
        unique_signals = set()
        for i in range(2 ** agent.bitN):
            meaning = torch.tensor(int2bin(agent.bitN, i), dtype=torch.float32).unsqueeze(0)
            signal = tuple(agent.m2s(meaning).round().squeeze(0).numpy().astype(int))
            unique_signals.add(signal)

    expressivity_score = len(unique_signals) / (2 ** agent.bitN)
    print(f"GETTING EXPRESSIVITY: {expressivity_score:.4f}")
    return expressivity_score



def hamming_distance(vec1, vec2):
    return sum(x != y for x, y in zip(vec1, vec2))


def compositionality(agent):
    meanings = [int2bin(agent.bitN, i) for i in range(2 ** agent.bitN)]

    with torch.no_grad():
        signals = [
            tuple(agent.m2s(torch.tensor(m, dtype=torch.float32).unsqueeze(0)).round().squeeze(0).numpy().astype(int))
            for m in meanings
        ]

    meaning_distances = []
    signal_distances = []

    # Compute pairwise distances
    for i in range(len(meanings)):
        for j in range(i + 1, len(meanings)):  # Avoid duplicate calculations
            d_m = hamming_distance(meanings[i], meanings[j])
            d_s = hamming_distance(signals[i], signals[j])

            meaning_distances.append(d_m)
            signal_distances.append(d_s)

    # Compute Spearman's correlation
    if len(set(meaning_distances)) == 1 or len(set(signal_distances)) == 1:
        return 0  # Avoid singular matrix error

    correlation, _ = spearmanr(meaning_distances, signal_distances)
    print(f"COMPOSITIONALITY: {correlation:.4f}")
    return correlation


def main():
    bitN = 8
    nodeN = 20
    batch_size = 30
    epochs = 100
    generations = 30

    stability_scores, expressivity_scores, compositionality_scores = iterated_learning(
        generations, bitN, nodeN, batch_size, epochs
    )

    # Plot the results
    plot_results(stability_scores, expressivity_scores, compositionality_scores, generations)


if __name__ == "__main__":
    #set params here
    main()
