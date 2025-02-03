import random as rn
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.utils import shuffle


class Agent:
    def __init__(self, bitN, m2s, s2m):
        self.bitN = bitN
        self.m2s = m2s  # Encoder- meaning → signal
        self.s2m = s2m  # Decoder - sibgnal → meaning
        self.m2m = nn.Sequential(m2s, s2m)  # Autoencoder


def create_agent(bitN, nodeN):
    m2s = nn.Sequential(
        nn.Linear(bitN, nodeN),  # meaning → hidden
        nn.Sigmoid(),  # activation
        nn.Linear(nodeN, bitN),  # hidden → signal
        nn.Sigmoid()
    )

    s2m = nn.Sequential(
        nn.Linear(bitN, nodeN),  # signal → hidden
        nn.Sigmoid(),  # activation
        nn.Linear(nodeN, bitN),  # hidden → meaning
        nn.Sigmoid()
    )
    return Agent(bitN, m2s, s2m)


def int2bin(bitN, value):
    return [int(x) for x in f"{value:0{bitN}b}"]


def generate_meaning_space(bitN):
    return [torch.tensor(int2bin(bitN, i), dtype=torch.float32) for i in range(2 ** bitN)]


def gen_supervised_data(agent, tutor, B_size, all_meanings):
    sampled_meanings = rn.sample(all_meanings, B_size)  # Randomly select B_size meanings
    m2sdata = []
    for meaning in sampled_meanings:
        signal = tutor.m2s(meaning.unsqueeze(0)).detach().round().squeeze(0)
        m2sdata.append((meaning.numpy(), signal.numpy()))

    s2mdata = m2sdata.copy()
    shuffle(s2mdata)

    return m2sdata, s2mdata


def gen_unsupervised_data(A_size, all_meanings):
    # all_meanings = generate_meaning_space(bitN)
    return rn.sample(all_meanings, A_size)  # Randomly select A_size meanings


### --- Training Function ---
def train_combined(agent, tutor, A_size, B_size, all_meanings, epochs, autoencoder_iters):
    optimiser = torch.optim.SGD(
        list(agent.m2s.parameters()) + list(agent.s2m.parameters()), lr=1.0
    )
    loss_function = nn.MSELoss()

    # Get supervised dataset (B_size) from tutor
    m2sdata, s2mdata = gen_supervised_data(agent, tutor, B_size, all_meanings)
    unsupervised_data = gen_unsupervised_data(A_size, all_meanings)
    # Get unsupervised meanings (A_size)

    for epoch in range(epochs):
        total_loss = 0

        # Select `batch_size` pairs per epoch from B_size dataset
        batch_indices = np.random.choice(len(m2sdata), B_size, replace=False)
        batch_m2s = [m2sdata[i] for i in batch_indices]
        batch_s2m = [s2mdata[i] for i in batch_indices]

        for (m2s_meaning, m2s_signal), (s2m_meaning, s2m_signal) in zip(batch_m2s, batch_s2m):
            m2s_meaning = torch.tensor(m2s_meaning, dtype=torch.float32).unsqueeze(0)
            m2s_signal = torch.tensor(m2s_signal, dtype=torch.float32).unsqueeze(0)

            s2m_meaning = torch.tensor(s2m_meaning, dtype=torch.float32).unsqueeze(0)
            s2m_signal = torch.tensor(s2m_signal, dtype=torch.float32).unsqueeze(0)

            optimiser.zero_grad()

            # Train encoder (m2s) and decoder (s2m)
            pred_m2s = agent.m2s(m2s_meaning)
            loss_m2s = loss_function(pred_m2s, m2s_signal)

            pred_s2m = agent.s2m(s2m_signal)
            loss_s2m = loss_function(pred_s2m, s2m_meaning)

            loss = loss_m2s + loss_s2m
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # Unsupervised Training (A-size)
        for _ in range(autoencoder_iters):
            # meanings_u = rn.sample(all_meanings, B_size)
            optimiser.zero_grad()
            pred_m2m = agent.m2m(torch.stack(unsupervised_data))
            loss_auto = loss_function(pred_m2m, torch.stack(unsupervised_data))
            loss_auto.backward()
            optimiser.step()

            total_loss += loss_auto.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


def iterated_learning(generations=20, bitN=10, nodeN=10, A_size=30, B_size=100, epochs=20):
    print("\n==== Initializing first tutor (random, untrained network) ====")
    tutor = create_agent(bitN, nodeN)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []
    all_meanings = generate_meaning_space(bitN)

    for gen in range(1, generations + 1):
        print(f"\n==== Generation {gen} ====")
        pupil = create_agent(bitN, nodeN)

        # Train using separate A and B sets
        train_combined(pupil, tutor, A_size, B_size, all_meanings, epochs, autoencoder_iters=30)

        stability_scores.append(stability(tutor, pupil, all_meanings))
        expressivity_scores.append(expressivity(pupil, all_meanings))
        compositionality_scores.append(compositionality(pupil))

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

### --- Evaluation Metrics ---
# Evaluation
def stability(tutor, pupil, all_meanings):
    tutor.m2s.eval()
    pupil.s2m.eval()

    with torch.no_grad():
        matches = 0
        for _ in range(len(all_meanings)):
            meaning = torch.randint(0, 2, (tutor.bitN,), dtype=torch.float32).unsqueeze(0)

            # Encode meaning using the tutor
            tutor_signal = tuple(tutor.m2s(meaning).round().squeeze(0).numpy().astype(int))

            # Decode signal using the pupil
            pupil_meaning = tuple(pupil.s2m(torch.tensor(tutor_signal, dtype=torch.float32).unsqueeze(0))
                                  .round().squeeze(0).numpy().astype(int))

            meaning_tuple = tuple(meaning.squeeze(0).numpy().astype(int))
            # print("===========----NEW----==============")
            # print(f"meaning tuple: {meaning_tuple}")
            # print(f"pupil_meaning: {pupil_meaning}")

            if meaning_tuple == pupil_meaning:
                matches += 1

    stability_score = matches / len(all_meanings)
    # print(f"GETTING STABILITY: {stability_score:.4f}")
    return stability_score



def expressivity(agent, all_meanings):
    agent.m2s.eval()
    with torch.no_grad():
        unique_signals = set()
        for meaning in all_meanings:
            print("===========----NEW----==============")
            # if len(all_meanings) == 0: print("Empty dataset")
            # meaning =
            print(f"expressivity_MEANING: {meaning}")
            signal = tuple(agent.m2s(meaning).round().squeeze(0).numpy().astype(int))
            print(f"expressivity_SIGNAL: {signal}")
            unique_signals.add(signal)

    # print(len(unique_signals))
    print(unique_signals)

    expressivity_score = len(unique_signals) / (2 ** agent.bitN)
    print(f"GETTING EXPRESSIVITY: {expressivity_score:.4f}")
    return expressivity_score


def calculate_entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compositionality(agent):
    meanings = [int2bin(agent.bitN, i) for i in range(2 ** agent.bitN)]

    with torch.no_grad():
        signals = [
            tuple(agent.m2s(torch.tensor(m, dtype=torch.float32).unsqueeze(0))
                  .round().squeeze(0).detach().numpy().astype(int))
            for m in meanings
        ]

    n = len(meanings)  # Number of meanings
    bit_length = len(meanings[0])  # Number of bits per meaning

    hij_matrix = np.zeros((n, bit_length))  # Stores entropy for each (meaning, signal) pair

    # Compute entropy for each meaning-signal relationship
    for i in range(n):
        for j in range(bit_length):
            count_1 = sum(1 for k in range(n) if meanings[k][j] == 1)
            p = count_1 / n  # Probability of '1' at bit position j
            hij_matrix[i, j] = calculate_entropy(p)

    # Step 1: Compute the minimal entropy per meaning
    h_i = np.min(hij_matrix, axis=1)

    # Step 2: Refine compositionality measure to prevent word overloading
    H = np.array([np.min(hij_matrix[:, j]) for j in range(bit_length)])
    h_prime = np.array([np.min(H) if np.any(H != 0) else 1 for _ in range(n)])

    # Compute final compositionality score
    c = 1 - (1 / n) * np.sum(h_prime)

    # print(f"COMPOSITIONALITY (Entropy-based): {c:.4f}")
    return c



### --- Main Function ---
def main():
    stability_scores, expressivity_scores, compositionality_scores = iterated_learning()
    plot_results(stability_scores, expressivity_scores, compositionality_scores, 20)


if __name__ == "__main__":
    main()