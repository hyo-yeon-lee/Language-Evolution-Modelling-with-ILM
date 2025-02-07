import random
import random as rn
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

#check how the weights are randomly initialised (check the range)
#check this with conor!
def create_agent(bitN, nodeN, i):
    m2s = nn.Sequential(
        nn.Linear(bitN, nodeN),  # meaning → hidden
        nn.Sigmoid(),  # activation
        nn.Linear(nodeN, bitN),  # hidden → signal
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


# def gen_unsupervised_data(A_size, all_meanings):
#     return rn.sample(all_meanings, A_size)

# choosing them with replacement for the unsupervised dataset : same meaning could come out twicel
def train_combined(agent, tutor, A_size, B_size, all_meanings, epochs):
    optimiser = torch.optim.SGD(
        list(agent.m2s.parameters()) + list(agent.s2m.parameters()), lr=5.0
    )
    # A IS UNSUPERVISED, B IS SUPERVISED TRAINING SET
    loss_function = nn.MSELoss()

    T = gen_supervised_data(tutor, all_meanings)
    # print(len(T))
    print(f"Current tutor gen: {tutor.num}")

    for epoch in range(epochs):
        total_loss = 0

        # Create B1 and B2 by selecting B_size pairs from T
        # random choice or samples?
        B1 = [random.choice(T) for _ in range(B_size)]
        B2 = B1.copy()
        A = [torch.tensor(meaning, dtype=torch.float32) for meaning, _ in B1]

        random.shuffle(B1)
        random.shuffle(B2)
        random.shuffle(A)

        # Train encoder (m2s) and decoder (s2m)
        for (m2s_meaning, m2s_signal), (s2m_meaning, s2m_signal) in zip(B1, B2):
            m2s_meaning = torch.tensor(m2s_meaning, dtype=torch.float32).unsqueeze(0)
            # print("===================NEW RANGE===================")
            # print(f"ms2 meaning: {m2s_meaning}")
            m2s_signal = torch.tensor(m2s_signal, dtype=torch.float32).unsqueeze(0)

            s2m_meaning = torch.tensor(s2m_meaning, dtype=torch.float32).unsqueeze(0)
            s2m_signal = torch.tensor(s2m_signal, dtype=torch.float32).unsqueeze(0)

            optimiser.zero_grad()

            pred_m2s = agent.m2s(m2s_meaning)
            loss_m2s = loss_function(pred_m2s, m2s_signal)

            pred_s2m = agent.s2m(s2m_signal)
            loss_s2m = loss_function(pred_s2m, s2m_meaning)

            loss = loss_m2s + loss_s2m
            loss.backward()
            optimiser.step()
            total_loss += loss.item()


        # Unsupervised Training (A-size)
            for _ in range(20):
                optimiser.zero_grad()
                meanings_u = rn.sample(A, 20)
                meanings_u_tensors = torch.stack(meanings_u)
                pred_m2m = agent.m2m(meanings_u_tensors)
                loss_auto = loss_function(pred_m2m, meanings_u_tensors)
                # print(f"loss in m2m: {loss_auto}")
                loss_auto.backward()
                optimiser.step()

                total_loss += loss_auto.item()

        # print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


def iterated_learning(generations=20, bitN=8, nodeN=8, A_size=75, B_size=75, epochs=20):
    print("\n==== Initializing first tutor (random, untrained network) ====")
    tutor = create_agent(bitN, nodeN, 1)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []
    all_meanings = generate_meaning_space(bitN)

    for gen in range(1, generations + 1):
        print(f"\n==== Generation {gen} ====")
        pupil = create_agent(bitN, nodeN, gen)
        print(f"Current pupil gen: {pupil.num}")

        # Train using separate A and B sets
        train_combined(pupil, tutor, A_size, B_size, all_meanings, epochs)

        stability_scores.append(stability(tutor, pupil, all_meanings))
        expressivity_scores.append(expressivity(pupil, all_meanings))
        compositionality_scores.append(compositionality(pupil, all_meanings))

        tutor = pupil
        print(f"Tutor gen: {tutor.num}")

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

    # Expressivity
    plt.subplot(1, 3, 2)
    plt.plot(gens, expressivity_scores, color="orange", linewidth=3)
    plt.xlabel("Generations")
    plt.ylabel("x")
    plt.title("B: Expressivity Over Generations", fontsize=16, loc="left")

    # Compositionalirty
    plt.subplot(1, 3, 3)
    plt.plot(gens, compositionality_scores, color="green", linewidth=3)
    plt.xlabel("Generations")
    plt.ylabel("c")
    plt.title("C: Compositionality Over Generations", fontsize=16, loc="left")

    plt.suptitle(rf"n = {len(stability_scores)}", fontsize=16, x=0.5, y=1.05)
    plt.tight_layout()
    plt.show()



def stability(tutor, pupil, all_meanings):
    tutor.m2s.eval()
    pupil.s2m.eval()

    matches = 0
    total_meanings = len(all_meanings)

    with torch.no_grad():
        for _ in range(total_meanings):
            # Generate a random binary meaning vector
            meaning = torch.randint(0, 2, (tutor.bitN,), dtype=torch.float32).unsqueeze(0)

            tutor_signal = tutor.m2s(meaning).round()  # Keep tensor format
            pupil_output = pupil.s2m(tutor_signal).round()

            # Convert tensors to tuples for comparison
            meaning_tuple = tuple(meaning.squeeze().cpu().detach().numpy().astype(int))
            pupil_meaning = tuple(pupil_output.squeeze().cpu().detach().numpy().astype(int))
            print("-----------NEW EPOCH===========")
            print(f"meaning tuple: {meaning_tuple}")
            print(f"pupil meaning: {pupil_meaning}")

            # Check if decoded meaning matches original meaning
            if meaning_tuple == pupil_meaning:
                matches += 1

    # Compute stability score
    stability_score = matches / total_meanings
    return stability_score


def expressivity(agent, all_meanings):
    agent.m2s.eval()
    with torch.no_grad():
        unique_signals = set()
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
    # meanings = [int2bin(agent.bitN, i) for i in range(2 ** agent.bitN)]

    with torch.no_grad():
        signals = [
            tuple(agent.m2s(torch.tensor(m, dtype=torch.float32).unsqueeze(0))
                  .round().squeeze(0).detach().numpy().astype(int))
            for m in all_meanings
        ]

    n = len(all_meanings)
    bit_length = agent.bitN

    hij_matrix = np.zeros((n, bit_length))

    for i in range(n):
        for j in range(bit_length):
            count_1 = sum(1 for k in range(n) if all_meanings[k][j] == 1)
            p = count_1 / n
            hij_matrix[i, j] = calculate_entropy(p)

    h_i = np.min(hij_matrix, axis=1)

    H = np.array([np.min(hij_matrix[:, j]) for j in range(bit_length)])

    h_prime = np.array([
        np.min([hij_matrix[i, j] for j in range(bit_length) if hij_matrix[i, j] in H])
        if np.any(H != 0) else 1
        for i in range(n)
    ])

    c = 1 - (1 / n) * np.sum(h_prime)

    return c


if __name__ == "__main__":
    stability_scores, expressivity_scores, compositionality_scores = iterated_learning()
    plot_results(stability_scores, expressivity_scores, compositionality_scores, 20)