import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class Agent:
    def __init__(self, bitN, m2s, s2m, i):
        self.bitN = bitN
        self.m2s = m2s  # Encoder- meaning → signal
        self.s2m = s2m  # Decoder - sibgnal → meaning
        self.m2m = nn.Sequential(m2s, s2m)  # Autoencoder
        self.num = i


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # 64 → 32
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 32 → 16
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 16 → 8
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 8 → 4
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

#check how the weights are randomly initialised (check the range) -> -0.3 ~ 0.3
def create_agent(bitN, nodeN, i):
    m2s = nn.Sequential(
        nn.Linear(bitN, nodeN),  # meaning → hidden
        nn.Sigmoid(),  #activation
        nn.Linear(nodeN, bitN),  # hidden → signal
        nn.Sigmoid()
    )

    s2m = nn.Sequential(
        nn.Linear(bitN, nodeN), # signal->hidden
        nn.Sigmoid(),
        nn.Linear(nodeN, bitN), # hidden -> meaning
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


# def check_initial_weights(agent):
#     print(f"Agent {agent.num} - Initial Weights Range:")
#
#     for name, param in agent.m2s.named_parameters():
#         if "weight" in name:  # Only check weight tensors (not biases)
#             print(f"  m2s {name} → min: {param.min().item()}, max: {param.max().item()}")
#
#     for name, param in agent.s2m.named_parameters():
#         if "weight" in name:
#             print(f"  s2m {name} → min: {param.min().item()}, max: {param.max().item()}")


def train_combined(agent, tutor, A_size, B_size, all_meanings, epochs):
    optimiser_m2s = torch.optim.SGD(agent.m2s.parameters(), lr=5.0)
    optimiser_s2m = torch.optim.SGD(agent.s2m.parameters(), lr=5.0)
    optimiser_m2m = torch.optim.SGD(list(agent.m2s.parameters()) + list(agent.s2m.parameters()), lr=5.0) # check if I'm using the optimiser c

    loss_function = nn.MSELoss()

    T = gen_supervised_data(tutor, all_meanings)
    A = gen_unsupervised_data(all_meanings, A_size)

    m2mtraining = 0

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
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
        pupil = create_agent(bitN, nodeN, gen)
        # Train using separate A and B sets
        train_combined(pupil, tutor, A_size, B_size, all_meanings, epochs)

        stability_scores.append(stability(tutor, pupil, all_meanings))
        expressivity_scores.append(expressivity(pupil, all_meanings))
        compositionality_scores.append(compositionality(pupil, all_meanings))

        tutor = pupil
        # print(f"Tutor gen: {tutor.num}")

    return stability_scores, expressivity_scores, compositionality_scores



def plot_results(stability_scores, expressivity_scores, compositionality_scores, generations, replicates=25):
    plt.figure(figsize=(15, 5))
    gens = np.arange(1, generations + 1)

    # Define colors
    colors = {'stability': 'purple', 'expressivity': 'blue', 'compositionality': 'orange',
        's': (0.5, 0.0, 0.5, 0.1),  # Light purple (RGBA with low alpha)
        'x': (0.0, 0.0, 1.0, 0.1),  # Light blue
        'c': (1.0, 0.65, 0.0, 0.1)  # Light orange
    }

    # Stability Plot
    plt.figure(figsize=(6, 4))
    for rep in stability_scores:
        plt.plot(gens, rep, color=colors['s'], alpha=0.2)
    plt.plot(gens, np.mean(stability_scores, axis=0), color=colors['stability'], linewidth=3)
    plt.xlabel("Generations", fontsize=12)
    plt.ylabel("s", fontsize=12)
    # plt.title("Stability Over Generations", fontsize=14)
    plt.show()

    # Expressivity Plot
    plt.figure(figsize=(6, 4))
    for rep in expressivity_scores:
        plt.plot(gens, rep, color=colors['x'], alpha=0.2)
    plt.plot(gens, np.mean(expressivity_scores, axis=0), color=colors['expressivity'], linewidth=3)
    plt.xlabel("Generations", fontsize=12)
    plt.ylabel("x", fontsize=12)
    # plt.title("Expressivity Over Generations", fontsize=14)
    plt.show()

    # Compositionality Plot
    plt.figure(figsize=(6, 4))
    for rep in compositionality_scores:
        plt.plot(gens, rep, color=colors['c'], alpha=0.2)
    plt.plot(gens, np.mean(compositionality_scores, axis=0), color=colors['compositionality'], linewidth=3)
    plt.xlabel("Generations", fontsize=12)
    plt.ylabel("c", fontsize=12)
    # plt.title("Compositionality Over Generations", fontsize=14)
    plt.show()




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
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compositionality(agent, all_meanings):
    n = agent.bitN
    num_messages = 2 ** n
    cnt = 0

    # Initialise matrices
    meaning_matrix = np.zeros((n, num_messages), dtype=int)
    signal_matrix = np.zeros((n, num_messages), dtype=int)

    for m in all_meanings:
        s = agent.m2s(m.unsqueeze(0)).detach().round().squeeze(0)
        meaning_matrix[:, cnt] = m
        signal_matrix[:, cnt] = s
        cnt += 1


    entropy = [[] for _ in range(n)]

    for m_col in range(n):
        curr_col_entropy = np.ones(n)
        for signal_col in range(n):
            p = np.sum(meaning_matrix[m_col, :] * signal_matrix[signal_col, :]) / (2 ** (n - 1))
            curr_col_entropy[signal_col] = calculate_entropy(p)

    # finding minimum entropy
        min_index = np.argmin(curr_col_entropy)
        min_val = curr_col_entropy[min_index]

        entropy[min_index].append(min_val)

    entropy_sum = sum(min(vals) if vals else 1 for vals in entropy)

    return 1 - entropy_sum / n


def main():
    generations = 50
    replicates = 25

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []

    for i in range(replicates):
        print(f"============================================{i}============================================")
        stability, expressivity, compositionality = iterated_learning(generations=generations)
        stability_scores.append(stability)
        expressivity_scores.append(expressivity)
        compositionality_scores.append(compositionality)

    plot_results(stability_scores, expressivity_scores, compositionality_scores, generations, replicates)


if __name__ == "__main__":
    main()
    # stability_scores, expressivity_scores, compositionality_scores = iterated_learning()
    # plot_results(stability_scores, expressivity_scores, compositionality_scores, 20)