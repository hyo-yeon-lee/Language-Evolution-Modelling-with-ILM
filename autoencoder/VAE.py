import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            nn.Linear(hid1, lat_dim * 2),
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
            nn.Sigmoid()  # Added sigmoid for pixel values between 0 and 1
        )

    def encode(self, x):
        encoded = self.m2s(x)
        mean, logvar = encoded.chunk(2, dim=1)
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
        reconst = self.decode(z)
        return reconst, mean, logvar


def loss_function(recon_x, x, mean, logvar, kl_weight):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss /= recon_x.shape[0]
    return recon_loss + kl_weight * kl_loss



def generate_meaning_space(shape, orientation, scale):
    dataset_zip = np.load(
        '/user/home/fw22912/diss/dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
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
    print(f"Filtered images: {filtered_imgs.shape}")

    # Convert to torch tensors
    return [torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in filtered_imgs]



def save_reconstructed_example(model, dataset, learning_rate):
    output_dir = f"/user/home/fw22912/diss/output_refined/lr_{learning_rate}"
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), 10)
    with torch.no_grad():
        for idx in indices:
            img = dataset[idx].unsqueeze(0).float().to(device)
            recon, _, _ = model(img)  # Now correctly unpacking three return values
            orig = img.squeeze().cpu().numpy()
            recon = recon.squeeze().cpu().numpy()

            fig, axs = plt.subplots(1, 2, figsize=(4, 2))
            axs[0].imshow(orig, cmap='gray')
            axs[0].set_title("Original Image")
            axs[0].axis('off')
            axs[1].imshow(recon, cmap='gray')
            axs[1].set_title("Reconstructed Image")
            axs[1].axis('off')
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"reconst_{idx}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved reconstructed image to {save_path}")


def train_vae(images, learning_rate):
    # Prepare dataset
    images_tensor = torch.stack(images)
    dataset = TensorDataset(images_tensor)
    print("Successfully loaded dataset...")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    print(f"Training set size: {len(train_set)}")
    print(f"Testing set size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    print("Training dataset loading complete!")
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    print("Testing dataset loading complete!")

    # Create model
    model = VAE(hid1=128, lat_dim=15)
    print("Successfully created a VAE model...")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    test_loss_history = []

    for epoch in range(150):
        model.train()
        total_train_loss = 0
        kl_weight = min(1.0, epoch / 50)  # gradually increase from 0 → 1 over 50 epochs

        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()

            # Corrected forward pass
            if isinstance(model, nn.DataParallel):
                recon, mean, logvar = model(x)
            else:
                recon, mean, logvar = model(x)

            print(f"Batch {epoch} recon range: {recon.min().item():.4f}-{recon.max().item():.4f}")
            loss = loss_function(recon, x, mean, logvar, kl_weight)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                recon, mean, logvar = model(x)
                # Corrected parameter names
                loss = loss_function(recon, x, mean, logvar, kl_weight)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/20 - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Rest of the code for plotting and saving reconstructions...
    output_dir = f"/user/home/fw22912/diss/output_refined/lr_{learning_rate}"
    os.makedirs(output_dir, exist_ok=True)

    # Plot train loss
    plt.figure(figsize=(6, 3))
    plt.plot(train_loss_history, label="Train Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"VAE Train Loss (LR={learning_rate})", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))
    plt.show()

    # Plot test loss
    plt.figure(figsize=(6, 3))
    plt.plot(test_loss_history, label="Test Loss", linewidth=2, color='orange')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"VAE Test Loss (LR={learning_rate})", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_loss.png"))
    plt.show()

    test_images = [test_set[i][0] for i in range(10)]
    test_tensor = torch.stack(test_images)
    save_reconstructed_example(model, test_tensor, learning_rate)

if __name__ == '__main__':
    shape = [1, 2]
    orientation = [10, 19, 29, 39]
    scale = [0, 1, 3, 5]
    all_meanings = generate_meaning_space(shape=shape, orientation=orientation, scale=scale)

    for lr in [5e-5, 5e-4, 1e-5]:
        print(f"\nTraining with learning rate: {lr}")
        train_vae(all_meanings, learning_rate=lr)
