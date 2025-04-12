import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


class Agent:
    def __init__(self, hid1, lat_dim, i):
        super(Agent, self).__init__()
        self.vae = VAE(hid1, lat_dim)
        self.num = i
        self.m2s = self.vae.m2s
        self.s2m = self.vae.s2m

    def m2m(self, x):
        mean, logvar = self.vae.encode(x)
        z = self.vae.reparameterise(mean, logvar)
        return self.vae.decode(z)


class VAE(nn.Module):
    def __init__(self, hid1, lat_dim):
        super(VAE, self).__init__()
        self.lat_dim = lat_dim

        # Encoder
        self.m2s = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, hid1),
            nn.ReLU(),
            nn.Linear(hid1, 256),
            nn.ReLU(),
            nn.Linear(256, lat_dim * 2),
        )

        # Decoder
        self.s2m = nn.Sequential(
            nn.Linear(lat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hid1),
            nn.ReLU(),
            nn.Linear(hid1, 8 * 8 * 128),  # Matches encoder's last feature map size
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),  # (batch, channels, height, width)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 → 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32x32 → 64x64
            nn.Sigmoid()
        )

    def encode(self, x):
        encoded = self.m2s(x)
        mean, logvar = encoded.chunk(2, dim=1)
        return mean, logvar

    def reparameterise(self, mean, logvar):
        logvar = torch.clamp(logvar, min=-4, max=4)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        sample = mean + std * epsilon
        return sample

    def decode(self, z):
        return self.s2m(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterise(mean, logvar)
        reconst = self.decode(z)
        return reconst, mean, logvar



def create_agent(hid1, lat_dim, i):
    return Agent(hid1, lat_dim, i)


def generate_meaning_space(shape, orientation, scale):
    dataset_zip = np.load(
        '/user/home/fw22912/diss/dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
        # '/Users/hyoyeon/Desktop/UNI/Year 3/Individual Project/Language-Evolution-Modelling-with-ILM/dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
        allow_pickle=True, encoding='latin1'
    )
    imgs = dataset_zip['imgs']
    latents_classes = dataset_zip['latents_classes']  # use discrete indices!

    mask = np.ones(len(imgs), dtype=bool)

    if shape is not None:
        mask &= np.isin(latents_classes[:, 1], shape)
    if orientation is not None:
        mask &= np.isin(latents_classes[:, 3], orientation)
    if scale is not None:
        mask &= np.isin(latents_classes[:, 2], scale)

    filtered_imgs = imgs[mask]
    filtered_latents = latents_classes[mask]
    # print(f"Filtered images: {filtered_imgs.shape}")

    meta_testset = [
        (int(latent[1]), int(latent[3]), int(latent[2]), torch.tensor(img, dtype=torch.float32).unsqueeze(0))
        for latent, img in zip(filtered_latents, filtered_imgs)
    ]

    return [torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in filtered_imgs], meta_testset


def gen_supervised_data(tutor, all_meanings):
    print("Generating supervised data...")
    T = []
    with torch.no_grad():  # optional but efficient
        for meaning in all_meanings:
            meaning = meaning.unsqueeze(0)  # shape: [1, 1, 64, 64]
            mean, logvar = tutor.vae.encode(meaning)
            signal = tutor.vae.reparameterise(mean, logvar).detach()
            T.append((meaning, signal))

            # print(f"Meaning shape: {meaning.shape}, Signal shape: {signal.shape}")
    return T


def gen_unsupervised_data(all_meanings, A_size):
    print("Generating unsupervised data...")
    U = []
    for _ in range(A_size):
        meaning = random.choice(all_meanings).unsqueeze(0)
        U.append(meaning)
    return U



def loss_function(recon_x, x, mean, logvar, lat_dim, beta):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = kl_loss / (recon_x.shape[0] * lat_dim)

    return recon_loss + beta * kl_loss



def train_combined(agent, tutor, A_size, B_size, all_meanings, epochs, beta, gen):
    print("Starting training...")
    optimiser_m2s = torch.optim.Adam(agent.m2s.parameters(), lr=1e-4)
    optimiser_s2m = torch.optim.Adam(agent.s2m.parameters(), lr=1e-4)
    optimiser_m2m = torch.optim.Adam(agent.vae.parameters(), lr = 1e-4)

    T = gen_supervised_data(tutor, all_meanings)
    A = gen_unsupervised_data(all_meanings, A_size)

    for epoch in range(epochs):
        print(f"\n==========================Gen: {gen}   Epoch {epoch + 1}/{epochs}==========================")
        B1 = [random.choice(T) for _ in range(B_size)]
        B2 = B1.copy()
        random.shuffle(B2)

        # print("Starting supervised training...")
        for i in range(B_size):
            optimiser_m2s.zero_grad()
            m2s_meaning, m2s_signal = B1[i]
            m2s_mean, m2s_logvar = agent.vae.encode(m2s_meaning)
            m2s_z = agent.vae.reparameterise(m2s_mean, m2s_logvar)
            # print(f"m2s_z signal shape: {m2s_z.shape}")

            loss_m2s = nn.MSELoss()(m2s_z, m2s_signal)
            loss_m2s.backward()
            optimiser_m2s.step()

            # training decoder
            optimiser_s2m.zero_grad()
            s2m_meaning, s2m_signal = B2[i]
            reconst_s2m = agent.vae.s2m(s2m_signal).to(torch.float32)

            loss_s2m = nn.MSELoss()(reconst_s2m, s2m_meaning)
            loss_s2m.backward()
            optimiser_s2m.step()

            # unsupervised training
            meanings_u = [random.choice(A) for _ in range(20)]
            # print("Starting unsupervised training...")
            for meaning in meanings_u:
                # print(f"unsupervised training meaning shape: {meaning.shape}")
                optimiser_m2m.zero_grad()
                mu, logvar = agent.vae.encode(meaning)
                reparam = agent.vae.reparameterise(mu, logvar)
                reconst = agent.vae.decode(reparam)

                loss_auto = loss_function(reconst, meaning, mu, logvar, lat_dim=5, beta=beta)
                loss_auto.backward()
                optimiser_m2m.step()



def random_test_sets(meta_dataset, sample_size):
    grouped = defaultdict(list)
    for item in meta_dataset:
        shape, orientation, scale, image_tensor = item
        grouped[(shape, orientation, scale)].append(item)

    samples_list = []
    for key, group_items in grouped.items():
        if len(group_items) >= sample_size:
            sampled_items = random.sample(group_items, sample_size)
        else:
            print(f"⚠ Warning: Not enough data for {key}, only {len(group_items)} available, taking all.")
            sampled_items = group_items
        samples_list.extend(sampled_items)

    count_check = defaultdict(int)
    for item in samples_list:
        key = (item[0], item[1], item[2])
        count_check[key] += 1

    for key, count in count_check.items():
        print(f"{key}: {count}")

    return samples_list



def iterated_learning(h_dim1, lat_dim, all_meanings, meta_testset, beta, generations=20,
                      A_size=300, B_size=300, epochs=20, sample_size=100):
    print("Entered iterated learning")
    tutor = create_agent(h_dim1, lat_dim, 1)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []

    for gen in range(1, generations + 1):
        print(f"================================================{gen}th GENERATION================================================")
        test_data = random_test_sets(meta_testset, sample_size)
        pupil = create_agent(h_dim1, lat_dim, gen)
        train_combined(pupil, tutor, A_size, B_size, all_meanings, epochs, beta, gen)

        stability_scores.append(stability(tutor, pupil, test_data))
        # expressivity_scores.append(expressivity(pupil, test_data))
        # compositionality_scores.append(compositionality(pupil, test_data))

        tutor = pupil

    # === Save original and reconstructed images from final agent ===
    print("\nSaving reconstructions from final agent...")

    save_dir = "/user/home/fw22912/diss/final_stability_reconst"
    os.makedirs(save_dir, exist_ok=True)

    final_agent = tutor  # last pupil
    final_agent.vae.eval()

    with torch.no_grad():
        for idx, item in enumerate(meta_testset):
            original = item[3].unsqueeze(0)  # shape: [1, 1, 64, 64]
            reconstructed = final_agent.m2m(original)

            # Combine side by side: [1, 1, 64, 64] → [1, 1, 64, 128]
            combined = torch.cat([original, reconstructed], dim=3)

            # Convert to numpy and save
            image_np = combined.squeeze().cpu().numpy()  # shape: [64, 128]
            plt.imsave(f"{save_dir}/comparison_{idx}.png", image_np, cmap='gray')

    return stability_scores, expressivity_scores, compositionality_scores


def plot_results(stability_scores, generations, replicates=25):
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
    plt.xlabel("Generations", fontsize=14)
    plt.ylabel("s", fontsize=14)

    save_dir = "/user/home/fw22912/diss/stability4020"
    os.makedirs(save_dir, exist_ok=True)

    filename = "stability.png"
    full_path = os.path.join(save_dir, filename)

    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")


    plt.show()

    # Expressivity Plot
    # plt.figure(figsize=(6, 4))
    # for rep in expressivity_scores:
    #     plt.plot(gens, rep, color=colors['x'], alpha=0.2)
    # plt.plot(gens, np.mean(expressivity_scores, axis=0), color=colors['expressivity'], linewidth=3)
    # plt.xlabel("Generations", fontsize=12)
    # plt.ylabel("x", fontsize=12)
    # # plt.title("Expressivity Over Generations", fontsize=14)
    # plt.show()

    # # Compositionality Plot
    # plt.figure(figsize=(6, 4))
    # for rep in compositionality_scores:
    #     plt.plot(gens, rep, color=colors['c'], alpha=0.2)
    # plt.plot(gens, np.mean(compositionality_scores, axis=0), color=colors['compositionality'], linewidth=3)
    # plt.xlabel("Generations", fontsize=12)
    # plt.ylabel("c", fontsize=12)
    # # plt.title("Compositionality Over Generations", fontsize=14)
    # plt.show()


def stability(tutor, pupil, all_meanings, mse_threshold=0.01):
    tutor.m2s.eval()
    pupil.s2m.eval()

    matches = 0
    total_meanings = len(all_meanings)

    with torch.no_grad():
        for item in all_meanings:
            meaning_tensor = item[3].unsqueeze(0)
            # print(f"meaning shape: {meaning_tensor.shape}")

            tut_mu, tut_logvar = tutor.vae.encode(meaning_tensor)
            tut_signal = tutor.vae.reparameterise(tut_mu, tut_logvar)
            pupil_s2m_mn = pupil.vae.decode(tut_signal)  # Note: calling decode on tutor's signal

            loss = nn.MSELoss()(meaning_tensor, pupil_s2m_mn)

            if loss < mse_threshold:
                matches += 1

    stability_score = matches / total_meanings
    print(f"stability score: {stability_score}")
    return stability_score




def expressivity(agent, all_meanings):
    agent.m2s.eval()
    unique_signals = set()

    with torch.no_grad():
        for meaning in all_meanings:
            meaning_tensor = meaning[3].clone().detach().unsqueeze(0).float()
            mu, logvar = agent.vae.encode(meaning_tensor)
            latent = agent.vae.reparameterise(mu, logvar)
            unique_signals.add(tuple(latent.squeeze(0).cpu().numpy()))

    expressivity_score = len(unique_signals) / len(all_meanings)
    print(f"Expressivity Score: {expressivity_score:.4f}")
    return expressivity_score



# def calculate_entropy(p):
#     if p == 0 or p == 1:
#         return 0
#     else:
#         return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
#
#
# def compositionality(agent, all_meanings, latent_dim, ):
#     n = agent.bitN
#     num_messages = 2 ** n
#     cnt = 0
#
#     # Initialise matrices
#     meaning_matrix = np.zeros((n, num_messages), dtype=int)
#     signal_matrix = np.zeros((n, num_messages), dtype=int)
#
#     for m in all_meanings:
#         s = agent.m2s(m.unsqueeze(0)).detach().round().squeeze(0)
#         meaning_matrix[:, cnt] = m
#         signal_matrix[:, cnt] = s
#         cnt += 1
#
#
#     entropy = [[] for _ in range(n)]
#
#     for m_col in range(n):
#         curr_col_entropy = np.ones(n)
#         for signal_col in range(n):
#             p = np.sum(meaning_matrix[m_col, :] * signal_matrix[signal_col, :]) / (2 ** (n - 1))
#             curr_col_entropy[signal_col] = calculate_entropy(p)
#
#     # finding minimum entropy
#         min_index = np.argmin(curr_col_entropy)
#         min_val = curr_col_entropy[min_index]
#
#         entropy[min_index].append(min_val)
#
#     entropy_sum = sum(min(vals) if vals else 1 for vals in entropy)
#
#     return 1 - entropy_sum / n


def main():
    # Desired values converted to indices:
    shape = [1, 2]
    orientation = [10, 19, 29, 39]
    scale = [0, 1, 3, 5]

    all_meanings, meta_testset = generate_meaning_space(shape=shape, orientation=orientation, scale=scale)
    generations = 20
    replicates = 1

    print("Starting iteration...")
    stability_scores = []
    # expressivity_scores = []
    # compositionality_scores = []

    for i in range(replicates):
        stability, expressivity, compositionality = iterated_learning(256, 5, all_meanings, meta_testset, beta=1e-3)
        stability_scores.append(stability)
    # expressivity_scores.append(expressivity)
    # compositionality_scores.append(compositionality)

    plot_results(stability_scores, generations, replicates)



# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--A_size", type=int, default=220)
#     parser.add_argument("--B_size", type=int, default=220)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--test_size", type=int, default=100)
#     parser.add_argument("--generations", type=int, default=20)
#     return parser.parse_args()


# def main():
#     args = parse_args()
#
#     shape = [0, 1]
#     orientation = [10, 19, 29, 39]
#     scale = [0, 1, 3, 5]
#
#     all_meanings, meta_testset = generate_meaning_space(shape=shape, orientation=orientation, scale=scale)
#
#     replicates = 1
#
#     stability_scores = []
#     expressivity_scores = []
#     compositionality_scores = []
#
#     for rep in range(replicates):
#         stability, expressivity, compositionality = iterated_learning(
#             h_dim1=128,
#             lat_dim=5,
#             all_meanings=all_meanings,
#             meta_testset=meta_testset,
#             generations=args.generations,
#             A_size=args.A_size,
#             B_size=args.B_size,
#             epochs=args.epochs,
#             etrain=20,
#             sample_size=args.test_size
#         )
#         stability_scores.append(stability)
#         expressivity_scores.append(expressivity)
#         compositionality_scores.append(compositionality)
#
#     plot_results(stability_scores, expressivity_scores, compositionality_scores, args.generations, replicates)



if __name__ == "__main__":
    main()
    # stability_scores, expressivity_scores, compositionality_scores = iterated_learning()
    # plot_results(stability_scores, expressivity_scores, compositionality_scores, 20)
