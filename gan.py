# implementation of GAN for simple minded people like me.
import typer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms # Import datasets and transforms
from torchvision.utils import make_grid
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from model import GeneratorConfigB as Generator, DiscriminatorConfigB as Discriminator # Assuming model.py now contains Generator and Discriminator
from soap import SOAP # Using cautious AdamW for cautious momentum
from pytorch_optimizer import get_wsd_schedule # Keeping scheduler import, though might not be directly applicable for GANs as WSD is for diffusion


class GAN:
    def __init__(self, generator, discriminator, latent_dim, device, num_classes):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device = device
        self.num_classes = num_classes
        self.r1_gamma = 1.0 # Example gamma for R1, adjust as needed
        self.r2_gamma = 1.0 # Example gamma for R2, adjust as needed

        # Load pre-trained VGG for perceptual loss monitoring
        self.vgg = models.vgg16(pretrained=True).features.eval().to(device) # Move VGG to device
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg_layers = ['conv3_3', 'conv4_3'] # Choose layers for perceptual loss
        self.vgg_outputs = {}
        def hook_fn(module, input, output, layer_name):
            self.vgg_outputs[layer_name] = output
        for name, module in self.vgg.named_modules():
            if name in ['7', '14']: # corresponding index for conv3_3 and conv4_3 in vgg16.features
                layer_name = 'conv' + name
                module.register_forward_hook(lambda module, input, output, layer_name=layer_name: hook_fn(module, input, output, layer_name))

    def generator_loss(self, fake_output, real_output):
        # RpGAN Generator loss with f(t) = -log(1 + e^-t) - non-saturating version
        return F.binary_cross_entropy_with_logits(fake_output - real_output, torch.ones_like(fake_output))

    def discriminator_loss(self, real_output, fake_output, real_images, fake_images):
        # RpGAN Discriminator loss with f(t) = -log(1 + e^-t) - non-saturating version
        real_loss = F.binary_cross_entropy_with_logits(real_output - fake_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy_with_logits(fake_output - real_output, torch.zeros_like(fake_output))
        r1_gp = self.r1_gradient_penalty(real_images, real_output) * self.r1_gamma # R1 regularization
        r2_gp = self.r2_gradient_penalty(fake_images, fake_output) * self.r2_gamma # R2 regularization
        return real_loss + fake_loss + r1_gp + r2_gp

    def r1_gradient_penalty(self, real_images, real_output):
        """R1 regularization gradient penalty."""
        grad_real = torch.autograd.grad(
            outputs=real_output.sum(), inputs=real_images, create_graph=True, retain_graph=True
        )[0]
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def r2_gradient_penalty(self, fake_images, fake_output):
        """R2 regularization gradient penalty on fake images."""
        # Create a detached copy of fake_images for R2 calculation
        fake_images_r2 = fake_images.detach().requires_grad_(True)
        # Use the detached copy for the forward pass in the discriminator
        fake_output_r2 = self.discriminator(fake_images_r2, torch.randint(0, self.num_classes, (fake_images.size(0),)).to(self.device)) # Assuming fake_labels are needed
        grad_fake = torch.autograd.grad(
            outputs=fake_output_r2.sum(), inputs=fake_images_r2, create_graph=True, retain_graph=True
        )[0]
        grad_penalty = grad_fake.pow(2).reshape(grad_fake.shape[0], -1).sum(1).mean()
        return grad_penalty

    def get_vgg_features(self, x):
        # Normalize input for VGG (assuming input is in [-1, 1])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        vgg_input = (x * 0.5 + 0.5) # [-1, 1] to [0, 1]
        if x.shape[1] == 1: # grayscale to rgb for vgg
            vgg_input = vgg_input.repeat(1, 3, 1, 1)
        vgg_input = (vgg_input - mean) / std
        _ = self.vgg(vgg_input) # forward pass to trigger hook functions
        features = [self.vgg_outputs['conv7'], self.vgg_outputs['conv14']] # Corresponding layers
        return features

    def perceptual_loss(self, generated_images, target_images):
        gen_vgg_features = self.get_vgg_features(generated_images)
        target_vgg_features = self.get_vgg_features(target_images)
        loss = 0.0
        for gen_feat, target_feat in zip(gen_vgg_features, target_vgg_features):
            loss += F.l1_loss(gen_feat, target_feat)
        return loss

    def forward(self, real_images, real_labels, optimizer_d, optimizer_g, g_scheduler, d_scheduler):
        batch_size = real_images.size(0)

        # Discriminator update
        optimizer_d.zero_grad()
        real_images.requires_grad_(True) # Ensure real_images requires grad for R1 regularization
        noise = torch.randn(batch_size, self.latent_dim).to(self.device) # Assuming latent_dim for generator input, adjusted for ConfigB
        fake_labels = torch.randint(0, self.num_classes, (batch_size,)).to(self.device) # Generate random labels for fake images
        fake_images = self.generator(noise, fake_labels) # Generate fake images with labels

        real_output = self.discriminator(real_images, real_labels) # Pass labels to discriminator
        fake_output = self.discriminator(fake_images.detach(), fake_labels) # Detach to not backprop through generator, pass fake labels
        disc_loss = self.discriminator_loss(real_output, fake_output, real_images, fake_images) # Pass fake_images without detach for R2
        disc_loss.backward()
        optimizer_d.step()
        d_scheduler.step()

        # Generator update
        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_labels = torch.randint(0, self.num_classes, (batch_size,)).to(self.device) # Generate random labels for generator
        fake_images_g = self.generator(noise, gen_labels) # Generate again for generator loss, pass gen_labels
        fake_output_g = self.discriminator(fake_images_g, gen_labels) # Pass gen_labels
        real_output_g = self.discriminator(real_images.detach(), real_labels) # Use detached real_images for generator update
        gen_loss = self.generator_loss(fake_output_g, real_output_g)
        gen_loss.backward()
        optimizer_g.step()
        g_scheduler.step()

        # MSE Loss Calculation (for monitoring, not training)
        mse_loss = F.mse_loss(fake_images_g.detach(), real_images.detach()).mean() # Calculate MSE, detach tensors

        # Perceptual Loss Calculation (for monitoring, not training)
        perceptual_loss_val = self.perceptual_loss(fake_images_g.detach(), real_images.detach()) # Calculate perceptual loss, detach

        return disc_loss, gen_loss, mse_loss, perceptual_loss_val, fake_images_g.detach() # Now return perceptual_loss


    @torch.no_grad()
    def sample(self, num_samples, labels):
        noise = torch.randn(num_samples, self.latent_dim).to(self.device) # Adjusted for ConfigB
        fake_images = self.generator(noise, labels) # Pass labels for sampling
        return fake_images


app = typer.Typer()

@app.command()
def main(
    cifar: bool = typer.Option(False, "--cifar", help="Use CIFAR-10 dataset"),
):
    """
    Train GAN model on MNIST or CIFAR-10.
    """

    CIFAR = cifar

    if CIFAR:
        dataset_name = "cifar_gan" # Different wandb project name
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        latent_dim = 128 # Example latent dimension
        generator = Generator(latent_dim=latent_dim, img_channels=channels, img_resolution=3) # Instantiate GeneratorConfigB
        discriminator = Discriminator(img_channels=channels, img_resolution=32) # Instantiate DiscriminatorConfigB


    else:
        dataset_name = "mnist_gan" # Different wandb project name
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        latent_dim = 32
        generator = Generator(latent_dim=latent_dim, img_channels=channels, img_resolution=32, base_channels=64) # Instantiate GeneratorConfigB
        discriminator = Discriminator(img_channels=channels, img_resolution=32, base_channels=64) # Instantiate DiscriminatorConfigB


    gen_size = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    disc_size = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {gen_size}, {gen_size / 1e6}M")
    print(f"Discriminator parameters: {disc_size}, {disc_size / 1e6}M")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = GAN(generator, discriminator, latent_dim, device, 10)

    # Optimizers
    g_optimizer = SOAP(gan.generator.parameters(), lr=3e-4, betas=(0.0, 0.9))
    d_optimizer = SOAP(gan.discriminator.parameters(), lr=3e-4, betas=(0.0, 0.9))
    
    steps = (60000 // 256) * 15 # Example steps calculation - adjust for GAN epochs/iterations
    g_scheduler = get_wsd_schedule(g_optimizer, steps * 0.05, steps * 0.6, steps * 0.35)
    d_scheduler = get_wsd_schedule(d_optimizer, steps * 0.05, steps * 0.6, steps * 0.35)

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=256, shuffle=True, drop_last=True)

    wandb.init(project=f"mnist") # Different wandb project name

    for epoch in range(15):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x, c) in pbar: # GANs usually don't use labels directly in basic setup
            x, c = x.to(device), c.to(device)
            disc_loss, gen_loss, mse_loss, perceptual_loss_val, fake_images = gan.forward(x, c, d_optimizer, g_optimizer, g_scheduler, d_scheduler) # Get perceptual_loss

            wandb.log({"discriminator_loss": disc_loss.item(), "generator_loss": gen_loss.item(), "mse_loss": mse_loss.item(), "perceptual_loss": perceptual_loss_val.item()}) # Log perceptual_loss
            pbar.set_description(f"Epoch {epoch+1}, D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}, MSE: {mse_loss.item():.4f}, VGG: {perceptual_loss_val.item():.4f}") # Show perceptual_loss in tqdm


            if i % (60000 // 2560) == 0:
                gan.generator.eval() # Set generator to eval mode for sampling
                sample_labels = torch.arange(0, 16) % 10 # Create labels for sampling
                sample_labels = sample_labels.to(device)
                samples = gan.sample(16, sample_labels) # Pass labels to sample
                # unnormalize
                image = samples * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                wandb.log({"sample": wandb.Image(img)})
                gan.generator.train() # Set back to train mode


        gan.generator.eval() # Set generator to eval mode for final epoch sample
        with torch.no_grad():
            sample_labels = torch.arange(0, 16) % 10 # Create labels for sampling
            sample_labels = sample_labels.to(device)
            samples = gan.sample(16, sample_labels) # Pass labels to sample
            # unnormalize
            image = samples * 0.5 + 0.5
            image = image.clamp(0, 1)
            x_as_image = make_grid(image.float(), nrow=4)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(f"contents/sample_gan_{epoch}_last.png") # Save last epoch sample
        gan.generator.train() # Set back to train mode


if __name__ == "__main__":
    app()