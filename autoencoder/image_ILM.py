import random
import numpy as np
import torch
import torch.nn as nn
from keras.src.ops import dtype
from matplotlib import pyplot as plt
from numpy.ma.core import shape
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
# from tensorflow.python.ops.numpy_ops.np_dtypes import float16
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Agent:
    def __init__(self, hid1, lat_dim, i):
        super(Agent, self).__init__()
        self.vae = VAE(hid1, lat_dim)
        self.num = i
        self.m2s = self.vae.m2s
        self.s2m = self.vae.s2m
        self.m2m = nn.Sequential(self.vae.m2s, self.vae.s2m)


class VAE(nn.Module):
    def __init__(self, hid1, lat_dim):
        super(VAE, self).__init__()
        self.lat_dim = lat_dim  # Store latent dimension

        # Encoder
        self.m2s = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),  # Flatten to 1D vector
            nn.Linear(8 * 8 * 128, hid1),
            nn.ReLU(),
            nn.Linear(hid1, lat_dim * 2)  # Outputs [mean, logvar]
        )

        # Decoder
        self.s2m = nn.Sequential(
            nn.Linear(lat_dim, hid1),
            nn.ReLU(),
            nn.Linear(hid1, 8 * 8 * 128),  # Use `lat_dim`, not fixed 16
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()
        )

    def encode(self, x):
        """Extracts mean and logvar from the encoder output"""
        encoded = self.m2s(x)  # Get [mean, logvar] combined
        mean, logvar = torch.chunk(encoded, 2, dim=1)  # Split tensor into 2 parts
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Samples from the learned distribution using the reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def decode(self, z):
        """Decodes latent vector back to image"""
        return self.s2m(z)

    def forward(self, x):
        """Passes input through the full VAE pipeline"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mean, logvar


def create_agent(hid1, lat_dim, i):
    return Agent(hid1, lat_dim, i)


def generate_meaning_space(shape, orientation, scale):
    # Load dataset
    dataset_zip = np.load('dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
    imgs = dataset_zip['imgs']  # Shape: (737280, 64, 64)
    # latents_classes = dataset_zip['latents_classes']
    latents_values = dataset_zip['latents_values']
    selected_indices = np.arange(len(imgs))

    print("Images shape:", dataset_zip['imgs'].shape)  # Should be (737280, 64, 64)

    if shape is not None:
        selected_indices = selected_indices[np.isin(latents_values[selected_indices, 1], shape)]

    if orientation is not None:
        selected_indices = selected_indices[np.isin(latents_values[selected_indices, 3], orientation)]

    if scale is not None:
        selected_indices = selected_indices[np.isin(latents_values[selected_indices, 2], scale)]

    filtered_imgs = imgs[selected_indices]
    print("Filtered shape:", filtered_imgs.shape)  # Should remain (N, 64, 64)
    return [torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in filtered_imgs]


def gen_supervised_data(tutor, all_meanings):
    print("Entered gen supervised data...")
    T = []
    for meaning in all_meanings:
        meaning = torch.tensor(meaning, dtype=torch.float16) if not isinstance(meaning, torch.Tensor) else meaning
        meaning = meaning.unsqueeze(0)  # Ensure batch dimension
        mean, logvar = tutor.vae.encode(meaning)
        signal = tutor.vae.reparameterize(mean, logvar).detach().squeeze(0)
        T.append((meaning.detach().cpu().numpy(), signal.detach().cpu().numpy()))

    return T



def gen_unsupervised_data(all_meanings, A_size):
    print("Entered gen unsupervised data...")
    U = []
    for _ in range(A_size):
        meaning = random.choice(all_meanings)
        U.append(meaning.numpy())
    return U



def loss_function(recon_x, x, mean, cov):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + cov - mean.pow(2) - cov.exp())
    loss = recon_loss + kl_loss
    return loss


def train_combined(agent, tutor, A_size, B_size, all_meanings, epochs):
    print("Entered train combined...")
    optimiser_m2s = torch.optim.Adam(agent.m2s.parameters(), lr=5.0)
    optimiser_s2m = torch.optim.Adam(agent.s2m.parameters(), lr=5.0)
    optimiser_m2m = torch.optim.Adam(list(agent.m2s.parameters()) + list(agent.s2m.parameters()),
                                     lr=5.0)  # check if I'm using the optimiser c

    # loss_function = nn.MSELoss()

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
            print(f"Processing {i+1}th data...")
            optimiser_m2s.zero_grad()
            m2s_meaning, m2s_signal = B1[i]
            m2s_meaning = torch.tensor(m2s_meaning, dtype=torch.float32)
            m2s_signal = torch.tensor(m2s_signal, dtype=torch.float32)
            # print(f"m2s_signal type: {np.shape(m2s_signal)}")

            m2s_mean, m2s_logvar = agent.vae.encode(m2s_meaning)
            m2s_z = agent.vae.reparameterize(m2s_mean, m2s_logvar).squeeze(0)
            loss_m2s = loss_function(m2s_z, m2s_signal, m2s_mean, m2s_logvar)
            loss_m2s.backward()
            optimiser_m2s.step()


            # training decoder
            optimiser_s2m.zero_grad()
            s2m_meaning, s2m_signal = B2[i]
            s2m_signal = torch.tensor(s2m_signal, dtype=torch.float32).unsqueeze(0)
            s2m_meaning = torch.tensor(s2m_meaning, dtype=torch.float32)
            print(f"s2m_signal shape before passing to decoder: {s2m_signal.shape}")

            pred_s2m = agent.vae.s2m(s2m_signal)
            pred_s2m = pred_s2m.to(torch.float32)
            print(pred_s2m.shape, s2m_meaning.shape)

            loss_s2m = nn.MSELoss()(pred_s2m, s2m_meaning)
            loss_s2m.backward()
            optimiser_s2m.step()

            print("----------finished training decoder-----------")


            # unsupervised training
            meanings_u = [random.choice(A) for _ in range(20)]
            for meaning in meanings_u:
                optimiser_m2m.zero_grad()
                auto_m = torch.tensor(meaning, dtype=torch.float32).unsqueeze(0)
                # print(f"auto_m shape: {auto_m}")
                m2s_z = agent.vae.reparameterize(m2s_mean, m2s_logvar).squeeze(0)
                # pred_m2m = agent.m2s(auto_m)
                mu, logvar = agent.vae.encode(auto_m)
                reparam = agent.vae.reparameterize(mu, logvar)
                reconst = agent.vae.decode(reparam)
                loss_auto = nn.MSELoss()(reconst, auto_m)
                loss_auto.backward()
                optimiser_m2m.step()

                m2mtraining += 1


def iterated_learning(h_dim1, lat_dim, all_meanings, generations=20, A_size=75, B_size=75, epochs=20):
    print("Entered iterated learning")
    tutor = create_agent(h_dim1, lat_dim, 1)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []
    test_sets = random.sample(all_meanings, 200)

    for gen in range(1, generations + 1):
        pupil = create_agent(h_dim1, lat_dim, gen)
        train_combined(pupil, tutor, A_size, B_size, all_meanings, epochs)

        stability_scores.append(stability(tutor, pupil, test_sets))
        expressivity_scores.append(expressivity(pupil, test_sets))
        compositionality_scores.append(compositionality(pupil, test_sets))

        tutor = pupil

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

    with torch.no_grad():
        tutor_latents = []
        pupil_latents = []

        for meaning in all_meanings:
            m = meaning.clone().detach().float().unsqueeze(0)
            encoded_m = pupil.m2s(m)
            pupil_latents.append(pupil.s2m(encoded_m).squeeze(0).numpy())

            tutor_encoded_m = tutor.m2s(m)  # Encode to latent space (1, 16)
            tutor_latents.append(tutor_encoded_m.squeeze(0).numpy())  # Store latent representation

    # Convert to numpy arrays
    tutor_latents = np.array(tutor_latents)
    pupil_latents = np.array(pupil_latents)

    # Run t-SNE on both generations
    tsne = TSNE(n_components=2, random_state=42)
    tutor_2d = tsne.fit_transform(tutor_latents)
    pupil_2d = tsne.fit_transform(pupil_latents)

    # Compute stability score based on similarity of representations
    similarity = np.mean([1 - cosine(t, p) for t, p in zip(tutor_2d, pupil_2d)])
    print(similarity)

    return similarity


def expressivity(agent, all_meanings):
    agent.m2s.eval()

    with torch.no_grad():
        latents = []
        for meaning in all_meanings:
            latents.append(agent.m2s(meaning.unsqueeze(0)).squeeze(0).numpy())

    # Convert to numpy
    latents = np.array(latents)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latents)

    # Compute dispersion (average pairwise Euclidean distance)
    distances = np.sqrt(np.sum((latent_2d[:, None, :] - latent_2d[None, :, :]) ** 2, axis=-1))
    expressivity_score = np.mean(distances)

    return expressivity_score  # Higher = more diverse meanings


def calculate_entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compositionality(agent, all_meanings, n_clusters=5):
    agent.m2s.eval()

    with torch.no_grad():
        latents = []
        for meaning in all_meanings:
            latents.append(agent.m2s(meaning.unsqueeze(0)).squeeze(0).numpy())

    # Convert to numpy
    latents = np.array(latents)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latents)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(latent_2d)

    # Compute cluster separation (higher separation = more compositional)
    cluster_separation = np.mean(
        [np.linalg.norm(latent_2d[i] - latent_2d[j]) for i in range(len(latent_2d)) for j in range(len(latent_2d)) if
         labels[i] == labels[j]])

    return cluster_separation  # Higher = more compositional


def main():
    shape = [2, 3]
    orientation = np.array([np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    scale = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    all_meanings = generate_meaning_space(shape=shape, orientation=orientation, scale=scale)
    generations = 50
    replicates = 25

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []

    # for i in range(replicates):
    #     print(f"============================================{i}============================================")
    stability, expressivity, compositionality = iterated_learning(256, 16, all_meanings)
    stability_scores.append(stability)
    # expressivity_scores.append(expressivity)
    # compositionality_scores.append(compositionality)
    #
    plot_results(stability_scores, expressivity_scores, compositionality_scores, generations, replicates)


if __name__ == "__main__":
    main()
    # stability_scores, expressivity_scores, compositionality_scores = iterated_learning()
    # plot_results(stability_scores, expressivity_scores, compositionality_scores, 20)
