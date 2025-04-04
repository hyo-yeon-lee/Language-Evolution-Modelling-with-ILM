import random
import torch
import torch.nn as nn
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
            nn.Flatten(),  # (128, 8, 8) = 8192
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



def loss_function(recon_x, x, mean, logvar, lat_dim, beta):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = kl_loss / (recon_x.shape[0] * lat_dim)
    print(
        f"Loss components: recon_loss = {recon_loss.item():.6f}, kl_loss = {kl_loss.item():.6f}")
    print(f"logvar: min={logvar.min().item():.4f}, max={logvar.max().item():.4f}")

    return recon_loss + beta * kl_loss



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
    # filtered_latents = latents_classes[mask]
    print(f"Filtered images: {filtered_imgs.shape}")

    return [torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in filtered_imgs]



def save_reconstructed_example(model, shape_vals, orientation_vals, scale_vals):
    output_dir = f"/user/home/fw22912/diss/full_lat7/reconstruction_grid"
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for s in shape_vals:
            for o in orientation_vals:
                for sc in scale_vals:
                    filtered_imgs = generate_meaning_space(shape=[s], orientation=[o], scale=[sc])
                    if not filtered_imgs:
                        print(f"Skipping shape={s}, orientation={o}, scale={sc} (no images)")
                        continue

                    img = filtered_imgs[0].unsqueeze(0).to(device)
                    recon, _, _ = model(img)
                    orig = img.squeeze().cpu().numpy()
                    recon = recon.squeeze().cpu().numpy()

                    fig, axs = plt.subplots(1, 2, figsize=(4, 2))
                    axs[0].imshow(orig, cmap='gray')
                    axs[0].set_title("Original")
                    axs[0].axis('off')
                    axs[1].imshow(recon, cmap='gray')
                    axs[1].set_title("Reconstructed")
                    axs[1].axis('off')
                    plt.tight_layout()

                    fname = f"reconst_s{s}_o{o}_sc{sc}.png"
                    save_path = os.path.join(output_dir, fname)
                    plt.savefig(save_path)
                    plt.close()
                    print(f"Saved: {save_path}")



def train_vae(images, learning_rate, target_beta, lat_dim):
    # Prepare dataset
    images_tensor = torch.stack(images)
    dataset = TensorDataset(images_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Create model
    model = VAE(hid1=256, lat_dim=lat_dim)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    test_loss_history = []

    for epoch in range(90):
        model.train()
        total_train_loss = 0

        # max_beta = 5e-2
        warmup_epochs = 40
        beta = min(target_beta, epoch / warmup_epochs * target_beta)

        for batch in train_loader:
            print(f"--------------------------------Epoch {epoch}/90--------------------------------")
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mean, logvar = model(x)

            loss = loss_function(recon, x, mean, logvar, lat_dim, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                loss = loss_function(recon, x, mean, logvar, lat_dim, beta)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/90 - Test Loss: {avg_test_loss:.4f}")


        # Log gradient norms
        model.eval()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"[Epoch {epoch}] Gradient norm: {total_norm:.4f}")

        # No extra optimizer.step() outside the batch loop

    # Rest of the code for plotting and saving reconstructions...
    output_dir = f"/user/home/fw22912/diss/full_lat7/lr_{beta}"
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
    # save_reconstructed_example(model, test_tensor, learning_rate, beta)
    save_reconstructed_example(
        model,
        shape_vals=shape,
        orientation_vals=orientation,
        scale_vals=scale
    )


if __name__ == '__main__':
    shape = [1, 2]
    orientation = [10, 19, 29, 39]
    scale = [0, 1, 3, 5]
    lat_dim = 7
    all_meanings = generate_meaning_space(shape=shape, orientation=orientation, scale=scale)

    for beta in [1e-3]:
        print(f"\nTraining with beta rate: {beta}")
        train_vae(all_meanings, learning_rate=1e-4, target_beta=beta, lat_dim=lat_dim)
