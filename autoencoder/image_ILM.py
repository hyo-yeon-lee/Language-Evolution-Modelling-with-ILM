import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class Agent:
    def __init__(self, org_dim, int_dim, lat_dim, i):
        # Instantiate the VAE with the provided dimensions
        self.vae = VAE(org_dim, int_dim, lat_dim, i)
        self.num = i
        self.m2s = self.vae.m2s   # encoder part
        self.s2m = self.vae.s2m   # decoder part
        self.m2m = nn.Sequential(self.vae.m2s, self.vae.s2m)  # autoencoder

    def forward(self, x):
        x_hat, z, mean, cov = self.vae.forward(x)
        return x_hat, z, mean, cov

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)



class VAE:
    def __init__(self, org_dim, int_dim, lat_dim, i):
        self.m2s = nn.Sequential(
            nn.Linear(org_dim, int_dim),
            nn.Sigmoid(),
            nn.Linear(int_dim, lat_dim),
            nn.Sigmoid()
        )

        self.mean = nn.Linear(lat_dim, 2)
        self.cov = nn.Linear(lat_dim, 2)

        self.s2m = nn.Sequential(
            nn.Linear(lat_dim, int_dim),
            nn.Sigmoid(),
            nn.Linear(int_dim, org_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.m2s(x)
        mean, cov = self.mean(x), self.cov(x)
        return mean, cov

    def decode(self, x):
        return self.s2m(x)

    def reparameterisation(self, mean, cov):
        dist = torch.distributions.Normal(0, cov)
        epsilon = dist.sample()
        z = mean + cov * epsilon
        return z

    def forward(self, x):
        mean, cov = self.encode(x)
        z = self.reparameterisation(mean, cov)
        x_hat = self.s2m(z)
        return x_hat, z, mean, cov



def create_agent(org_dim, int_dim, lat_dim, i):
    return Agent(org_dim, int_dim, lat_dim, i)


def img2bin(img, i):
    return None


def generate_meaning_space(bitN):
    dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']  # continuous values for latent factor
    latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]  # dataset details
    print('Metadata: \n', metadata)
    return [torch.tensor(img2bin(bitN, i), dtype=torch.float32) for i in range(2 ** bitN)]


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
    optimiser_m2s = torch.optim.Adam(agent.m2s.parameters(), lr=5.0)
    optimiser_s2m = torch.optim.Adam(agent.s2m.parameters(), lr=5.0)
    optimiser_m2m = torch.optim.Adam(list(agent.m2s.parameters()) + list(agent.s2m.parameters()), lr=5.0) # check if I'm using the optimiser c

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



def iterated_learning(x_dim, h_dim1, h_dim2, img_set, generations=20, A_size=75, B_size=75, epochs=20):
    tutor = create_agent(x_dim, h_dim1, h_dim2, 1)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []
    # all_meanings = generate_meaning_space(bitN)
    all_meanings = img_set

    for gen in range(1, generations + 1):
        pupil = create_agent(gen)
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