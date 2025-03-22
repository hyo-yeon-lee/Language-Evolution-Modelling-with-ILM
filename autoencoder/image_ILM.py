import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


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
        # print(f"shape: {x.shape}")
        encoded = self.m2s(x)
        encoded = encoded.view(encoded.shape[0], -1, 2)
        mean, logvar = encoded[:, :, 0], encoded[:, :, 1]
        return mean, logvar

    def reparameterise(self, mean, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        sample = mean + std * epsilon
        return sample

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
    dataset_zip = np.load(
        'dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
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
    print(f"Filtered images: {filtered_imgs.shape}")

    # Convert to torch tensors
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
                                     lr=0.001)  # check if I'm using the optimiser c

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
                m2s_z = agent.vae.reparameterise(m2s_mean, m2s_logvar).round().squeeze(0)
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
                auto_m = torch.tensor(meaning, dtype=torch.float32).unsqueeze(0)

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


def visualize_clustering(all_meanings, tsne_embeddings, clusters, n_clusters=32):
    # Extract original and reconstructed embeddings
    original_embeds = tsne_embeddings[:len(all_meanings)]
    reconstructed_embeds = tsne_embeddings[len(all_meanings):]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    scatter_orig = ax.scatter(
        original_embeds[:, 0], original_embeds[:, 1], original_embeds[:, 2],
        c=clusters, cmap='tab20', alpha=0.9, marker='o', label='Original Images'
    )

    # Plot reconstructed points with same color coding but different marker
    scatter_reconst = ax.scatter(
        reconstructed_embeds[:, 0], reconstructed_embeds[:, 1], reconstructed_embeds[:, 2],
        c=clusters, cmap='tab20', alpha=0.4, marker='x', label='Reconstructed Images'
    )

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.set_title('t-SNE Clustering: Original vs. Reconstructed Images')

    # Create legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='x', color='w', label='Reconstruction', markerfacecolor='gray', markeredgecolor='gray', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='best')

    # Add color bar to show cluster IDs
    cbar = plt.colorbar(scatter_orig, ax=ax, fraction=0.02, pad=0.1)
    cbar.set_label('Cluster ID')

    plt.show()




def iterated_learning(h_dim1, lat_dim, all_meanings, generations=20, A_size=75, B_size=75, epochs=20, etrain = 3):
    print("Entered iterated learning")
    tutor = create_agent(h_dim1, lat_dim, 1)

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []
    test_sets = random.sample(all_meanings, 3200)

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
    # n_clusters = len(shape) * len(orientation) * len(scale)
    n_clusters = 32

    orig_images = []
    reconst_images = []
    tutor.vae.eval()
    pupil.vae.eval()

    with torch.no_grad():
        for meaning in all_images:
            meaning_tensor = meaning.clone().detach().unsqueeze(0).float()
            orig_images.append(meaning_tensor.cpu().numpy().flatten())
            mu, logvar = tutor.vae.encode(meaning_tensor)
            z = tutor.vae.reparameterise(mu, logvar)

            reconstruction = pupil.vae.decode(z).cpu().numpy().squeeze(0)
            reconst_images.append(reconstruction.flatten())

    orig_images = np.array(orig_images)
    reconst_images = np.array(reconst_images)
    combined = np.vstack((orig_images, reconst_images))

    tsne = TSNE(n_components=3, perplexity=40, random_state=42)
    embeddings = tsne.fit_transform(combined)

    original_embeds = embeddings[:len(all_images)]
    reconstructed_embeds = embeddings[len(all_images):]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(original_embeds)

    distances = np.linalg.norm(reconstructed_embeds[:, None] - kmeans.cluster_centers_, axis=2)
    assigned_clusters = np.argmin(distances, axis=1)

    for i in range(5):
        dist_to_assigned = distances[i, assigned_clusters[i]]
        dist_to_true_cluster = distances[i, clusters[i]]
        print(
            f"Point {i}: dist to assigned cluster center = {dist_to_assigned:.4f}, dist to original cluster center = {dist_to_true_cluster:.4f}")

    stability_matches = (clusters == assigned_clusters)
    # print(f"Stability matches: {stability_matches}")
    print(f"Stabiltiy matches: {np.sum(stability_matches)}")

    stability_score = np.sum(stability_matches)/ len(all_images)

    visualize_clustering(all_images, embeddings, clusters, n_clusters=32)
    print(f"Image-based conceptual stability score (cluster agreement): {stability_score:.4f}")
    return stability_score



def expressivity(agent, all_meanings):
    agent.m2s.eval()
    unique_signals = set()

    with torch.no_grad():
        for meaning in all_meanings:
            meaning = torch.tensor(meaning, dtype=torch.float32).unsqueeze(0)
            mu, logvar = agent.vae.encode(meaning)
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
    shape = [0, 1]
    orientation = [10, 19, 29, 39]
    scale = [0, 1, 3, 5]

    all_meanings = generate_meaning_space(shape=shape, orientation=orientation, scale=scale)
    generations = 20
    replicates = 1

    stability_scores = []
    expressivity_scores = []
    compositionality_scores = []

    for i in range(replicates):
        #     print(f"============================================{i}============================================")
        stability, expressivity, compositionality = iterated_learning(128, 5, all_meanings) #just seeing whether it could capture the space
        stability_scores.append(stability)
    # expressivity_scores.append(expressivity)
    # compositionality_scores.append(compositionality)

    plot_results(stability_scores, expressivity_scores, compositionality_scores, generations, replicates)


if __name__ == "__main__":
    main()
    # stability_scores, expressivity_scores, compositionality_scores = iterated_learning()
    # plot_results(stability_scores, expressivity_scores, compositionality_scores, 20)
