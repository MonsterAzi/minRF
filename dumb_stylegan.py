import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
import math

# Configuration
latent_dim = 64
image_size = 28
image_channels = 1
base_channels = 64  # Configurable base channel size
batch_size = 256
learning_rate = 0.0002
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "simplified_stylegan2_mnist_output"
os.makedirs(output_dir, exist_ok=True)

# Bilinear Interpolation Resize
def bilinear_upsample(x, scale_factor=2):
    return nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)

def bilinear_downsample(x, scale_factor=2):
    return nn.functional.interpolate(x, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False):
        super(ResidualBlock, self).__init__()
        self.upsample = upsample
        self.downsample = downsample

        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        if downsample:
            layers.append(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)) # Using Upsample for downsampling
        self.residual_block = nn.Sequential(*layers)

        self.channel_adjust = None
        if in_channels != out_channels or upsample or downsample:
            adjust_layers = []
            if upsample:
                adjust_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            adjust_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
            if downsample:
                adjust_layers.append(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))
            self.channel_adjust = nn.Sequential(*adjust_layers)


    def forward(self, x):
        residual = x
        if self.channel_adjust:
            residual = self.channel_adjust(residual)
        return self.residual_block(x) + residual


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, base_channels, image_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_channels = image_channels

        self.initial_constant = nn.Parameter(torch.randn(1, base_channels * 4, 4, 4)) # Start from constant 4x4

        self.blocks = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 2, upsample=True), # 8x8
            ResidualBlock(base_channels * 2, base_channels, upsample=True),      # 16x16
            ResidualBlock(base_channels, base_channels // 2, upsample=True),   # 32x32, but we need 28x28, so will adjust later
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_channels // 2, image_channels, kernel_size=3, padding=1)
        )

    def forward(self, z):
        x = self.initial_constant.repeat(z.shape[0], 1, 1, 1) # Repeat constant for batch size
        x = self.blocks(x)
        x = bilinear_downsample(x, scale_factor=32/28) # Resize 32x32 to 28x28
        img = self.to_rgb(x)
        return img # No tanh


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, base_channels, image_channels):
        super(Discriminator, self).__init__()
        self.base_channels = base_channels
        self.image_channels = image_channels

        self.from_rgb = nn.Conv2d(image_channels, base_channels // 2, kernel_size=1) # 1x1 conv to get to base channels

        self.blocks = nn.Sequential(
            ResidualBlock(base_channels // 2, base_channels),        # 28x28
            ResidualBlock(base_channels, base_channels * 2, downsample=True),   # 14x14
            ResidualBlock(base_channels * 2, base_channels * 4, downsample=True),   # 7x7
        )

        self.final_conv = nn.Conv2d(base_channels * 4, 1, kernel_size=4, padding=0) # Output a single value

    def forward(self, img):
        x = self.from_rgb(img)
        x = self.blocks(x)
        x = self.final_conv(x) # [B, 1, 1, 1]
        return x.view(-1) # [B]


# Initialize models
generator = Generator(latent_dim, base_channels, image_channels).to(device)
discriminator = Discriminator(base_channels, image_channels).to(device)

# Optimizers (Adam with beta1=0)
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0, 0.99))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0, 0.99))

# Loss function (BCE with logits)
criterion = nn.BCEWithLogitsLoss()

# R1 Regularization
def r1_regularization(real_pred, real_img):
    """R1 regularization for discriminator."""
    grad_real = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0] # added create_graph=True
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

# Data loading (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(data_loader):
        real_images = real_images.to(device)
        batch_size_current = real_images.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_d.zero_grad()

        # Real images
        real_images.requires_grad_(True) # Ensure gradients are tracked for real_images
        real_pred = discriminator(real_images)
        real_loss = criterion(real_pred, torch.ones_like(real_pred))
        r1_penalty = r1_regularization(real_pred, real_images)
        d_real_loss = real_loss + r1_penalty * 10.0 # R1 regularization weight = 10

        # Fake images
        z = torch.randn(batch_size_current, latent_dim).to(device)
        fake_images = generator(z)
        fake_pred = discriminator(fake_images.detach()) # Detach to not train G here
        fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
        d_fake_loss = fake_loss

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_d.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_g.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size_current, latent_dim).to(device)
        fake_images = generator(z)
        fake_pred_g = discriminator(fake_images) # No detach here, train G to fool D
        g_loss = criterion(fake_pred_g, torch.ones_like(fake_pred_g))

        g_loss.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(data_loader)}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    # Save generated images
    if epoch % 5 == 0:
        z_sample = torch.randn(64, latent_dim).to(device)
        fake_images_sample = generator(z_sample)
        save_image(fake_images_sample * 0.5 + 0.5, os.path.join(output_dir, f"epoch_{epoch}.png"), nrow=8) # Scale back to [0, 1] for saving
        print(f"Saved generated images at epoch {epoch}")

print("Training finished!")