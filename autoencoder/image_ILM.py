import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


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
        # self.lat_dim = lat_dim

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
            nn.Linear(hid1, lat_dim * 2),
            # nn.Sigmoid()
        )

        # Decoder
        self.s2m = nn.Sequential(
            nn.Linear(lat_dim, hid1),
            nn.ReLU(),
            nn.Linear(hid1, 8 * 8 * 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ReLU()
        )

    def encode(self, x):
        encoded = self.m2s(x)
        encoded = encoded.view(encoded.shape[0], -1, 2)
        mean, logvar = encoded[:, :, 0], encoded[:, :, 1]
        return mean, logvar

    def reparameterise(self, mean, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def decode(self, z):
        return self.s2m(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterise(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mean, logvar


def create_agent(hid1, lat_dim, i):
    return Agent(hid1, lat_dim, i)


def generate_meaning_space(shape, orientation, scale):
    dataset_zip = np.load('dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']

    mask = np.ones(len(imgs), dtype=bool)  # Start with all True
    if shape is not None:
        mask &= np.isin(latents_values[:, 1], shape)
    if orientation is not None:
        mask &= np.isin(latents_values[:, 3], orientation)
    if scale is not None:
        mask &= np.isin(latents_values[:, 2], scale)

    filtered_imgs = imgs[mask]
    print(f"filtered images: {filtered_imgs.shape}")
    return [torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in filtered_imgs]



def gen_supervised_data(tutor, all_meanings):
    print("Entered gen supervised data...")
    T = []
    for meaning in all_meanings:
        meaning = torch.tensor(meaning, dtype=torch.float16) if not isinstance(meaning, torch.Tensor) else meaning
        meaning = meaning.unsqueeze(0)
        mean, logvar = tutor.vae.encode(meaning)
        signal = tutor.vae.reparameterise(mean, logvar).detach().round().squeeze(0)
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
    kl_loss /= recon_x.shape[0]  # Normalize by batch size only
    loss = recon_loss + kl_loss
    return loss


def train_combined(agent, tutor, A_size, B_size, all_meanings, epochs, etrain):
    print("Entered train combined...")
    optimiser_m2s = torch.optim.SGD(agent.m2s.parameters(), lr=0.001)
    optimiser_s2m = torch.optim.SGD(agent.s2m.parameters(), lr=0.001)
    optimiser_m2m = torch.optim.SGD(list(agent.m2s.parameters()) + list(agent.s2m.parameters()),
                                     lr=0.1)  # check if I'm using the optimiser c

    T = gen_supervised_data(tutor, all_meanings)
    A = gen_unsupervised_data(all_meanings, A_size)

    m2mtraining = 0
    for epoch in range(epochs):
        print(f"\n========================== Epoch {epoch + 1}/{epochs}==========================")
        B1 = [random.choice(T) for _ in range(B_size)]
        B2 = B1.copy()
        random.shuffle(B2)

        # print(f"\n============ Supervised Training ===============")
        for i in range(B_size):
            # training encoder
            # print(f"Processing {i+1}th data...")
            m2mtraining = 0
            optimiser_m2s.zero_grad()
            m2s_meaning, m2s_signal = B1[i]
            m2s_meaning = torch.tensor(m2s_meaning, dtype=torch.float32)
            m2s_signal = torch.tensor(m2s_signal, dtype=torch.float32)

            #multiple sampling from same distribution
            for e in range(etrain):
                m2s_mean, m2s_logvar = agent.vae.encode(m2s_meaning)
                m2s_z = agent.vae.reparameterise(m2s_mean, m2s_logvar).squeeze(0)
                # print("M2S Latent vector z:", m2s_z)
                loss_m2s = loss_function(m2s_z, m2s_signal, m2s_mean, m2s_logvar)
                loss_m2s.backward()
                optimiser_m2s.step()

            # training decoder
            optimiser_s2m.zero_grad()
            s2m_meaning, s2m_signal = B2[i]
            s2m_signal = torch.tensor(s2m_signal, dtype=torch.float32).unsqueeze(0)
            s2m_meaning = torch.tensor(s2m_meaning, dtype=torch.float32)

            pred_s2m = agent.vae.s2m(s2m_signal)
            pred_s2m = pred_s2m.to(torch.float32)
            # print(f"nan count in s2m: {np.count_nonzero(np.isnan(s2m_signal))}")
            # print(pred_s2m.shape, s2m_meaning.shape)

            loss_s2m = nn.MSELoss()(pred_s2m, s2m_meaning)
            loss_s2m.backward()
            optimiser_s2m.step()


            # unsupervised training
            meanings_u = [random.choice(A) for _ in range(20)]
            for meaning in meanings_u:
                # print(f"\n============ Unsupervised Training ===============")
                optimiser_m2m.zero_grad()
                auto_m = torch.tensor(meaning, dtype=torch.float32).round().unsqueeze(0)

                mu, logvar = agent.vae.encode(auto_m)
                reparam = agent.vae.reparameterise(mu, logvar)
                reconst = agent.vae.decode(reparam)

                loss_auto = nn.MSELoss()(reconst, auto_m)
                loss_auto.backward()
                optimiser_m2m.step()

                original_img = auto_m.squeeze().detach().cpu().numpy()
                reconstructed_img = reconst.squeeze().detach().cpu().numpy()
                m2mtraining += 1

                if i == 119 and m2mtraining == 20 and epoch == 39:
                    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                    axes[0].imshow(original_img, cmap='gray')
                    axes[0].set_title("Original Image")
                    axes[0].axis("off")

                    axes[1].imshow(reconstructed_img, cmap='gray')
                    axes[1].set_title("Reconstructed Image")
                    axes[1].axis("off")

                    plt.show()



def iterated_learning(h_dim1, lat_dim, all_meanings, generations=20, A_size=120, B_size=120, epochs=40, etrain = 5):
    print("Entered iterated learning")
    tutor = create_agent(h_dim1, lat_dim, 1)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []
    test_sets = random.sample(all_meanings, 200)

    for gen in range(1, generations + 1):
        print(f"================================================{gen}th GENERATION================================================")
        pupil = create_agent(h_dim1, lat_dim, gen)
        train_combined(pupil, tutor, A_size, B_size, all_meanings, epochs, etrain)

        stability_scores.append(stability(tutor, pupil, test_sets))
        expressivity_scores.append(expressivity(pupil, test_sets))
        # compositionality_scores.append(compositionality(pupil, test_sets))

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

    # # Compositionality Plot
    # plt.figure(figsize=(6, 4))
    # for rep in compositionality_scores:
    #     plt.plot(gens, rep, color=colors['c'], alpha=0.2)
    # plt.plot(gens, np.mean(compositionality_scores, axis=0), color=colors['compositionality'], linewidth=3)
    # plt.xlabel("Generations", fontsize=12)
    # plt.ylabel("c", fontsize=12)
    # # plt.title("Compositionality Over Generations", fontsize=14)
    # plt.show()


def stability(tutor, pupil, all_images):
    tutor.m2s.eval()
    pupil.s2m.eval()

    matches = 0
    total_images = len(all_images)

    with torch.no_grad():
        for image in all_images:
            img = image.clone().detach().float().unsqueeze(0)
            mu, logvar = tutor.vae.encode(img)
            latent = tutor.vae.reparameterise(mu, logvar)
            reconstructed_img = pupil.vae.decode(latent)
            loss = nn.MSELoss()(reconstructed_img, img)
            # total_loss += loss.item() 
            print(f"loss: {loss}")

            if loss == 0 : #maybe I could set a proper threshold.
                print("Matches!")
                matches += 1

    stability_score = matches / total_images
    print(f"Stability Score: {stability_score:.4f}")
    return stability_score



def expressivity(agent, all_meanings):
    agent.m2s.eval()
    unique_signals = set()

    with torch.no_grad():
        for meaning in all_meanings:
            mu, logvar = agent.vae.encode(meaning)
            latent = agent.vae.reparameterise(mu, logvar)
            unique_signals.add(tuple(latent.squeeze(0).cpu().numpy()))

    expressivity_score = len(unique_signals) / len(all_meanings)
    return expressivity_score



def calculate_entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# def compositionality(agent, all_meanings, n_clusters=5):
#     agent.m2s.eval()
#
#     with torch.no_grad():
#         latents = []
#         for meaning in all_meanings:
#             latents.append(agent.m2s(meaning.unsqueeze(0)).squeeze(0).numpy())
#
#     # Convert to numpy
#     latents = np.array(latents)
#
#     # Run t-SNE
#     tsne = TSNE(n_components=2, random_state=42)
#     latent_2d = tsne.fit_transform(latents)
#
#     # Apply KMeans clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(latent_2d)
#
#     # Compute cluster separation (higher separation = more compositional)
#     cluster_separation = np.mean(
#         [np.linalg.norm(latent_2d[i] - latent_2d[j]) for i in range(len(latent_2d)) for j in range(len(latent_2d)) if
#          labels[i] == labels[j]])
#
#     return cluster_separation  # Higher = more compositional


def main():
    shape = [3]
    orientation = np.array([2 * np.pi])
    scale = np.array([1. ])
    all_meanings = generate_meaning_space(shape=shape, orientation=orientation, scale=scale)
    generations = 50
    replicates = 25

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []

    # for i in range(replicates):
    #     print(f"============================================{i}============================================")
    stability, expressivity, compositionality = iterated_learning(128, 12, all_meanings) #just seeing whether it could capture the space
    stability_scores.append(stability)
    # expressivity_scores.append(expressivity)
    # compositionality_scores.append(compositionality)

    plot_results(stability_scores, expressivity_scores, compositionality_scores, generations, replicates)


if __name__ == "__main__":
    main()
    # stability_scores, expressivity_scores, compositionality_scores = iterated_learning()
    # plot_results(stability_scores, expressivity_scores, compositionality_scores, 20)
