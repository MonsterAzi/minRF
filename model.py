import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample

        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)) # Bias True as no normalization
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if downsample:
            layers.append(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)) # Upsample with scale_factor=0.5 for downsampling

        self.residual_block = nn.Sequential(*layers)

        self.shortcut = nn.Sequential() # Initialize as empty sequential by default
        if in_channels != out_channels or upsample or downsample:
            shortcut_layers = []
            if upsample:
                shortcut_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            if in_channels != out_channels:
                shortcut_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)) # Bias True as no normalization
            if downsample:
                shortcut_layers.append(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)) # Upsample with scale_factor=0.5 for downsampling
            self.shortcut = nn.Sequential(*shortcut_layers)


    def forward(self, x):
        residual = self.residual_block(x)
        shortcut = self.shortcut(x)
        return residual + shortcut


class GeneratorConfigB(nn.Module):
    def __init__(self, latent_dim, img_resolution=256, img_channels=3, base_channels=0, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.num_resolutions = int(torch.log2(torch.tensor(img_resolution)).item()) - 1 # e.g., 256 -> 7 resolutions after 4x4

        flag = False
        if base_channels < 1:
            base_channels = img_resolution
            flag = True
        
        self.base_channels = base_channels

        current_channels = base_channels # Initial channels after constant input

        self.initial_constant = nn.Parameter(torch.randn(1, base_channels, 4, 4)) # 4x4 constant input
        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, latent_dim)
            self.latent_projection = nn.Linear(latent_dim * 2, base_channels * 16)
        else:
            self.latent_projection = nn.Linear(latent_dim, base_channels * 16)

        self.blocks = nn.ModuleList()
        self.to_rgb_convs = nn.ModuleList() # List to store ToRGB convs
        self.blocks.append(ResidualBlock(current_channels, current_channels))
        self.blocks.append(ResidualBlock(current_channels, current_channels))
        self.to_rgb_convs.append(nn.Conv2d(current_channels, img_channels, kernel_size=1, padding=0, bias=False)) # ToRGB for 4x4
        
        next_channels = current_channels
        
        for res_idx in range(self.num_resolutions - 2, -1, -1): # From resolution 4x4 up to img_resolution
            if flag:
                next_channels = current_channels // 2
            # Two residual blocks per resolution
            self.blocks.append(ResidualBlock(current_channels, current_channels, upsample=True)) # Upsample in the first block
            self.blocks.append(ResidualBlock(current_channels, next_channels)) # No upsample in the second block
            self.to_rgb_convs.append(nn.Conv2d(next_channels, img_channels, kernel_size=1, padding=0, bias=False)) # ToRGB for each resolution
            current_channels = next_channels


    def forward(self, z, c):
        c_emb = self.class_embedding(c)
        z = torch.cat([z, c_emb], dim=1)
        z = self.latent_projection(z)
        z = z.view(-1, self.base_channels, 4, 4)
        x = z * self.initial_constant.repeat(z.size(0), 1, 1, 1) # Constant input

        output_image = None # Initialize accumulated image

        block_idx = 0
        to_rgb_idx = 0
        for _ in range(self.num_resolutions): # Iterate through resolutions
            x = self.blocks[block_idx](x)
            block_idx += 1
            x = self.blocks[block_idx](x)
            block_idx += 1

            intermediate_rgb = self.to_rgb_convs[to_rgb_idx](x) # Apply ToRGB conv
            to_rgb_idx += 1

            if output_image is None: # First resolution (4x4)
                output_image = intermediate_rgb
            else:
                output_image = F.interpolate(output_image, scale_factor=2, mode='bilinear', align_corners=False) + intermediate_rgb

        return output_image


class DiscriminatorConfigB(nn.Module):
    def __init__(self, img_resolution=256, img_channels=3, base_channels=0, num_classes=10):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.num_resolutions = int(torch.log2(torch.tensor(img_resolution)).item()) - 1
        
        flag = False
        if base_channels < 1:
            base_channels = 4
            flag = True
        
        current_channels = base_channels
        
        self.base_channels = base_channels

        self.input_conv = nn.Conv2d(img_channels, current_channels, kernel_size=1, padding=0, bias=True) # Bias True as no normalization
        self.blocks = nn.ModuleList()

        next_channels = current_channels

        for res_idx in range(1, self.num_resolutions): # From resolution img_resolution down to 8x8
            if flag:
                next_channels = current_channels * 2
            # Two residual blocks per resolution
            self.blocks.append(ResidualBlock(current_channels, next_channels)) # No downsample in the first block
            self.blocks.append(ResidualBlock(next_channels, next_channels, downsample=True)) # Downsample in the second block
            current_channels = next_channels
        
        self.blocks.append(ResidualBlock(current_channels, current_channels))
        self.blocks.append(ResidualBlock(current_channels, current_channels))

        self.flatten_conv = nn.Conv2d(current_channels, current_channels, kernel_size=4, bias=True) # Flatten to 1x1 after conv
        self.feature_layer = nn.Linear(current_channels, current_channels)
        self.class_embedding = nn.Embedding(num_classes, current_channels)


    def forward(self, img, c):
        x = self.input_conv(img)

        for block in self.blocks:
            x = block(x)

        x = self.flatten_conv(x)
        x = x.view(x.size(0), -1) # Flatten
        features = self.feature_layer(x)
        c_emb = self.class_embedding(c)
        output = torch.sum(c_emb * features, dim=1, keepdim=True)
        return output


if __name__ == '__main__':
    latent_dim = 64
    resolution = 32 # Reduced resolution for faster testing, original was 256
    batch_size = 8
    num_classes = 10 # Define number of classes

    G = GeneratorConfigB(latent_dim, img_resolution=resolution, num_classes=num_classes, base_channels=32) # Pass num_classes
    D = DiscriminatorConfigB(img_resolution=resolution, img_channels=3, num_classes=num_classes, base_channels=32) # Pass num_classes

    z = torch.randn(batch_size, latent_dim)
    labels_g = torch.randint(0, num_classes, (batch_size,)) # Generate random labels for generator
    fake_img = G(z, labels_g) # Pass labels to generator
    assert fake_img.shape == (batch_size, 3, resolution, resolution)

    real_img = torch.randn(batch_size, 3, resolution, resolution)
    labels_d_fake = torch.randint(0, num_classes, (batch_size,)) # Generate random labels for discriminator (fake)
    labels_d_real = torch.randint(0, num_classes, (batch_size,)) # Generate random labels for discriminator (real)
    fake_output = D(fake_img, labels_d_fake) # Pass labels to discriminator (fake)
    real_output = D(real_img, labels_d_real) # Pass labels to discriminator (real)

    print("Generator output shape:", fake_img.shape)
    print("Discriminator fake output shape:", fake_output.shape)
    print("Discriminator real output shape:", real_output.shape)
    print("Generator parameters:", sum(p.numel() for p in G.parameters() if p.requires_grad))
    print("Discriminator parameters:", sum(p.numel() for p in D.parameters() if p.requires_grad))