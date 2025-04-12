import os
import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class Agent:
    def __init__(self, bitN, m2s, s2m, i):
        self.bitN = bitN
        self.m2s = m2s
        self.s2m = s2m
        self.m2m = nn.Sequential(m2s, s2m)
        self.num = i


def create_agent(bitN, nodeN, i):
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
    return Agent(bitN, m2s, s2m, i)


def int2bin(bitN, value):
    return [int(x) for x in f"{value:0{bitN}b}"]


def generate_meaning_space(bitN):
    return [torch.tensor(int2bin(bitN, i), dtype=torch.float32) for i in range(2 ** bitN)]


def gen_supervised_data(tutor, all_meanings):
    T = []
    for meaning in all_meanings:
        signal = tutor.m2s(meaning.unsqueeze(0)).detach().round().squeeze(0)
        T.append((meaning.numpy(), signal.numpy()))
    return T


def gen_unsupervised_data(all_meanings, A_size):
    U = []
    for _ in range(A_size):
        meaning = random.choice(all_meanings)
        U.append(meaning.numpy())
    return U


def train_combined(agent, tutor, A_size, B_size, all_meanings, epochs):
    optimiser_m2s = torch.optim.SGD(agent.m2s.parameters(), lr=5.0)
    optimiser_s2m = torch.optim.SGD(agent.s2m.parameters(), lr=5.0)
    optimiser_m2m = torch.optim.SGD(list(agent.m2s.parameters()) + list(agent.s2m.parameters()), lr=5.0)

    loss_function = nn.MSELoss()

    T = gen_supervised_data(tutor, all_meanings)
    A = gen_unsupervised_data(all_meanings, A_size)

    m2mtraining = 0

    for epoch in range(epochs):
        B1 = [random.choice(T) for _ in range(B_size)]
        B2 = B1.copy()
        random.shuffle(B2)

        for i in range(B_size):
            # training encoder
            optimiser_m2s.zero_grad()
            m2s_meaning, m2s_signal = B1[i]
            m2s_meaning = torch.tensor(m2s_meaning, dtype=torch.float32).unsqueeze(0)
            m2s_signal = torch.tensor(m2s_signal, dtype=torch.float32).unsqueeze(0)

            pred_m2s = agent.m2s(m2s_meaning)
            loss_m2s = loss_function(pred_m2s, m2s_signal)
            loss_m2s.backward()
            optimiser_m2s.step()

            # training decoder
            optimiser_s2m.zero_grad()
            s2m_meaning, s2m_signal = B2[i]
            s2m_signal = torch.tensor(s2m_signal, dtype=torch.float32).unsqueeze(0)
            s2m_meaning = torch.tensor(s2m_meaning, dtype=torch.float32).unsqueeze(0)

            pred_s2m = agent.s2m(s2m_signal)
            loss_s2m = loss_function(pred_s2m, s2m_meaning)
            loss_s2m.backward()
            optimiser_s2m.step()

            # unsupervised training
            meanings_u = [random.choice(A) for _ in range(20)]
            for meaning in meanings_u:
                optimiser_m2m.zero_grad()
                auto_m = torch.tensor(meaning, dtype=torch.float32).unsqueeze(0)

                pred_m2m = agent.m2m(auto_m)
                loss_auto = loss_function(pred_m2m, auto_m)
                loss_auto.backward()
                optimiser_m2m.step()

                m2mtraining += 1



def iterated_learning(generations=20, bitN=8, nodeN=8, A_size=75, B_size=75, epochs=20):
    tutor = create_agent(bitN, nodeN, 1)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []
    all_meanings = generate_meaning_space(bitN)

    for gen in range(1, generations + 1):
        print(f"generation: {gen}")
        pupil = create_agent(bitN, nodeN, gen)
        train_combined(pupil, tutor, A_size, B_size, all_meanings, epochs)

        stability_scores.append(stability(tutor, pupil, all_meanings))
        expressivity_scores.append(expressivity(pupil, all_meanings))
        compositionality_scores.append(compositionality(pupil, all_meanings))

        tutor = pupil

    return stability_scores, expressivity_scores, compositionality_scores



def plot_results(stability_scores, expressivity_scores, compositionality_scores, generations):
    plt.figure(figsize=(15, 5))
    gens = np.arange(1, generations + 1)

    save_path = "your_path"

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    colors = {'stability': 'purple', 'expressivity': 'blue', 'compositionality': 'orange',
        's': (0.5, 0.0, 0.5, 0.1),
        'x': (0.0, 0.0, 1.0, 0.1),
        'c': (1.0, 0.65, 0.0, 0.1)
    }

    # Stability Plot
    plt.figure(figsize=(6, 4))
    for rep in stability_scores:
        plt.plot(gens, rep, color=colors['s'], alpha=0.2)
    plt.plot(gens, np.mean(stability_scores, axis=0), color=colors['stability'], linewidth=4)
    plt.xlabel("Generations", fontsize=13)
    plt.ylabel("s", fontsize=14)
    # plt.show()

    if save_path is not None:
        stability_file = os.path.join(save_path, "stability.png")
        plt.savefig(stability_file, dpi=300)
    # plt.show()

    # Expressivity Plot
    plt.figure(figsize=(6, 4))
    for rep in expressivity_scores:
        plt.plot(gens, rep, color=colors['x'], alpha=0.2)
    plt.plot(gens, np.mean(expressivity_scores, axis=0), color=colors['expressivity'], linewidth=4)
    plt.xlabel("Generations", fontsize=13)
    plt.ylabel("x", fontsize=14)
    # plt.show()

    if save_path is not None:
        expressivity_file = os.path.join(save_path, "expressivity.png")
        plt.savefig(expressivity_file, dpi=300)
    # plt.show()


    # Compositionality Plot
    plt.figure(figsize=(6, 4))
    for rep in compositionality_scores:
        plt.plot(gens, rep, color=colors['c'], alpha=0.2)
    plt.plot(gens, np.mean(compositionality_scores, axis=0), color=colors['compositionality'], linewidth=4)
    plt.xlabel("Generations", fontsize=13)
    plt.ylabel("c", fontsize=14)
    # plt.show()

    if save_path is not None:
        compositionality_file = os.path.join(save_path, "compositionality.png")
        plt.savefig(compositionality_file, dpi=300)
    # plt.show()



def stability(tutor, pupil, all_meanings):
    tutor.m2s.eval()
    pupil.s2m.eval()

    matches = 0
    total_meanings = len(all_meanings)

    with torch.no_grad():
        for meaning in all_meanings:
            m = meaning.clone().detach().float().unsqueeze(0)
            tutor_m2s_sig = tutor.m2s(m)
            pupil_s2m_mn = pupil.s2m(tutor_m2s_sig)

            original_arr = meaning.numpy() > 0.5
            decoded_arr = pupil_s2m_mn.squeeze(0).numpy() > 0.5

            if np.array_equal(original_arr, decoded_arr):
                matches += 1

    stability_score = matches / total_meanings
    return stability_score



def expressivity(agent, all_meanings):
    agent.m2s.eval()
    unique_signals = set()

    with torch.no_grad():
        for meaning in all_meanings:
            signal = tuple(agent.m2s(meaning).round().squeeze(0).numpy().astype(int))
            unique_signals.add(signal)

    expressivity_score = len(unique_signals) / (2 ** agent.bitN)
    return expressivity_score


def calculate_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compositionality(agent, all_meanings):
    n = agent.bitN
    num_messages = 2 ** n

    meaning_matrix = np.zeros((n, num_messages), dtype=int)
    signal_matrix = np.zeros((n, num_messages), dtype=int)

    cnt = 0

    for m in all_meanings:
        s = agent.m2s(m.unsqueeze(0)).detach().round().squeeze(0)
        meaning_matrix[:, cnt] = m.numpy()
        signal_matrix[:, cnt] = s.numpy()
        cnt += 1

    #  minimal entropy calculation
    fact_min_entropies = np.zeros(n)  # h_i for each fact i
    fact_best_word = np.zeros(n, dtype=int)  # j index that minimizes h_ij for fact i

    for i in range(n):
        min_entropy = np.inf
        best_j = -1
        for j in range(n):
            p = np.sum(meaning_matrix[i, :] * signal_matrix[j, :]) / 2 ** (n - 1)
            h_ij = calculate_entropy(p)

            if h_ij < min_entropy:
                min_entropy = h_ij
                best_j = j
        fact_min_entropies[i] = min_entropy
        fact_best_word[i] = best_j

    # Collision resolution
    adjusted_entropies = fact_min_entropies.copy()
    for j in range(n):
        facts_using_j = np.where(fact_best_word == j)[0]
        if len(facts_using_j) > 1:
            best_fact = facts_using_j[np.argmin(fact_min_entropies[facts_using_j])]

            for idx in facts_using_j: #penalyt
                if idx != best_fact:
                    adjusted_entropies[idx] = 1.0

    # compute compositionality score
    average_adjusted_entropy = np.mean(adjusted_entropies)
    compositionality_score = 1 - average_adjusted_entropy

    return compositionality_score


def main():
    generations = 50
    replicates = 25

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []

    for i in range(replicates):
        print(f"Replicates: {i}")
        stability, expressivity, compositionality = iterated_learning(generations=generations)
        stability_scores.append(stability)
        expressivity_scores.append(expressivity)
        compositionality_scores.append(compositionality)

    plot_results(stability_scores, expressivity_scores, compositionality_scores, generations)


if __name__ == "__main__":
    main()
