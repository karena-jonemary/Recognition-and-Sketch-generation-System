import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from core.networks import UNetGenerator, NLayerDiscriminator, PerceptualLoss
from core.dataset import MaskedFaceDataset

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    generator = UNetGenerator(in_channels=5, out_channels=3).to(device)
    discriminator = NLayerDiscriminator(input_nc=3).to(device)
    perceptual_loss = PerceptualLoss().to(device)

    # Initialize optimizers
    g_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    dataset = MaskedFaceDataset(args.data_dir, img_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print(f"Starting training on {device}...")
    for epoch in range(args.epochs):
        g_losses, d_losses = [], []
        for i, batch in enumerate(loader):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_opt.zero_grad()
            
            # Real
            pred_real = discriminator(targets)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # Fake
            fake_imgs = generator(inputs)
            pred_fake = discriminator(fake_imgs.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            d_opt.step()

            # -----------------
            #  Train Generator
            # -----------------
            g_opt.zero_grad()
            
            # GAN Loss
            pred_fake = discriminator(fake_imgs)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            
            # L1 Loss (focus more on masked region)
            loss_g_l1_unmasked = criterion_l1(fake_imgs * (1 - masks), targets * (1 - masks))
            loss_g_l1_masked = criterion_l1(fake_imgs * masks, targets * masks)
            loss_g_l1 = loss_g_l1_unmasked + 5.0 * loss_g_l1_masked
            
            # Perceptual Loss
            loss_g_vgg = perceptual_loss(fake_imgs, targets)
            
            # Total Loss
            loss_g = loss_g_gan + 100.0 * loss_g_l1 + 10.0 * loss_g_vgg
            loss_g.backward()
            g_opt.step()

            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())

        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss D: {sum(d_losses)/len(d_losses):.4f}, Loss G: {sum(g_losses)/len(g_losses):.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(args.save_dir, f"generator_ep{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.save_dir, f"discriminator_ep{epoch+1}.pth"))

if __name__ == "__main__":
    train()
