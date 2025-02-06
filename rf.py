# implementation of Rectified Flow for simple minded people like me.
import typer

import torch
import torch.nn.functional as F
from torchvision import models, datasets, transforms # Import datasets and transforms
from torchvision.utils import make_grid
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from dit import DiUT_Llama
from pytorch_optimizer import get_wsd_schedule, OrthoGrad
from soap import SOAP


class RF:
    def __init__(self, model, ln=True, bias=-3.1, perceptual_loss_weight=0.005):
        self.model = model
        self.ln = ln
        self.bias = bias
        self.perceptual_loss_weight = perceptual_loss_weight

        # Load pre-trained VGG for perceptual loss
        self.vgg = models.vgg16(pretrained=True).features.eval().cuda()
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


    def sigmoidal_weighting(self, logsnr):
        return torch.sigmoid(logsnr - self.bias)

    def MSE(self, error):
        return error ** 2

    def MAE(self, error):
        return torch.abs(error)

    def log_cosh(self, error):
        return torch.log(torch.cosh(error))

    def velocity_loss(self, truth, sample):
        return torch.cosine_similarity(truth, sample, dim=1, eps=1e-10)

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

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        lc_loss = self.MSE(z1 - x - vtheta)

        # logsnr bias
        logsnr = 2 * torch.log((1 - t) / t)
        logsnr = logsnr.clamp(-20, 20)
        weights = self.sigmoidal_weighting(logsnr)

        batchwise_loss = weights * lc_loss.mean(dim=list(range(1, len(x.shape))))

        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]

        # Perceptual Loss
        v_target = z1 - x
        v_predict = vtheta
        perceptual_loss = self.perceptual_loss(v_predict, v_target)

        total_loss = batchwise_loss.mean()

        return total_loss, ttloss, batchwise_mse.mean(), perceptual_loss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images


app = typer.Typer()

@app.command()
def main(
    cifar: bool = typer.Option(False, "--cifar", help="Use CIFAR-10 dataset"),
):
    """
    Train Rectified Flow model on MNIST or CIFAR-10.
    """

    CIFAR = cifar

    if CIFAR:
        dataset_name = "cifar"
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
        model = DiUT_Llama(
            channels, 32, dim=64, n_steps=1, n_layers=6, n_heads=4, num_classes=10, num_moe_experts=1
        ).cuda()

    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        model = DiUT_Llama(
            channels, 32, dim=64, n_steps=1, n_layers=4, n_heads=4, num_classes=10, num_moe_experts=1
        ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    # Create the optimizer
    optimizer = SOAP(model.parameters(), lr=1e-2, betas=(.95, .95), weight_decay=.001, precondition_frequency=4)
    steps = (60000 // 192) * 15
    scheduler = get_wsd_schedule(optimizer, steps * 0.01, steps * 0.64, steps * 0.35)

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=192, shuffle=True, drop_last=True)

    wandb.init(project=f"rf_{dataset_name}")

    for epoch in range(15):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x, c) in pbar:
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct, mse_loss, perceptual_loss = rf.forward(x, c) # Get perceptual loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({"loss": loss.item(), "mse_loss": mse_loss.item(), "perceptual_loss": perceptual_loss.item()}) # Log perceptual loss
            pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, Perceptual: {perceptual_loss.item():.4f}")


            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

            if i % (60000 // 2560) == 0:
                rf.model.eval()
                with torch.no_grad():
                    cond = torch.arange(0, 16).cuda() % 10
                    uncond = torch.ones_like(cond) * 10

                    init_noise = torch.randn(16, channels, 32, 32).cuda()
                    images = rf.sample(init_noise, cond, uncond, sample_steps=50)
                    # unnormalize
                    image = images[-1] * 0.5 + 0.5
                    image = image.clamp(0, 1)
                    x_as_image = make_grid(image.float(), nrow=4)
                    img = x_as_image.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    wandb.log({"sample": wandb.Image(img)})
                rf.model.train()

        # log
        for i in range(10):
            print(f"Epoch: {epoch+1}, {i} range loss: {lossbin[i] / losscnt[i]}")

        wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % 10
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, channels, 32, 32).cuda()
            images = rf.sample(init_noise, cond, uncond)
            # image sequences to gif
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"contents/sample_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(f"contents/sample_{epoch}_last.png")

        rf.model.train()


if __name__ == "__main__":
    app()