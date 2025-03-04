from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os
from PIL import Image
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddim import DDIM

# Custom dataset for AID
class AIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Get all image paths from all subdirectories
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith('.jpg') or img_name.endswith('.png'):
                        self.image_paths.append(os.path.join(class_path, img_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as label since we don't care about classes


def train_aid(
    n_epoch: int = 100, 
    device: str = "cuda:2", 
    load_pth: Optional[str] = None,
    img_size: int = 256,  # Resize images to this size
    batch_size: int = 32,  # Smaller batch size due to larger images
    lr: float = 1e-5,     # Learning rate
    use_augmentation: bool = True,  # Whether to use data augmentation
    n_feat: int = 256     # Number of base features in UNet
) -> None:

    # Create a UNet with appropriate capacity for the dataset
    # Using n_feat=256 which is the default in NaiveUnet
    # This provides more capacity for the complex satellite imagery
    sampler = DDIM(eps_model=NaiveUnet(3, 3, n_feat=n_feat), betas=(1e-4, 0.02), eta=0.5, n_T=1000)

    if load_pth is not None:
        sampler.load_state_dict(torch.load(load_pth))

    sampler.to(device)

    # Transform with resize to make training more manageable
    # Adding data augmentation for better generalization
    if use_augmentation:
        tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Create AID dataset
    dataset = AIDDataset(
        root_dir="./data/AID/data",
        transform=tf,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    optim = torch.optim.Adam(sampler.parameters(), lr=lr)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Create output directory if it doesn't exist
    os.makedirs("./contents/aid", exist_ok=True)

    best_loss = float('inf')
    
    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        sampler.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        epoch_losses = []
        
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = sampler(x)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=1.0)
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            
            epoch_losses.append(loss.item())
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # Calculate average loss for the epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {i} average loss: {avg_epoch_loss:.4f}")
        
        # Update learning rate based on loss
        scheduler.step(avg_epoch_loss)

        sampler.eval()
        with torch.no_grad():
            # Sample images with the same dimensions as our training data
            xh = sampler.sample(8, (3, img_size, img_size), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/aid/sampler_sample_aid_{i}.png")

            # Save model checkpoint
            torch.save(sampler.state_dict(), f"./sampler_aid.pth")
            
            # Save a checkpoint every 10 epochs
            if i % 10 == 0:
                torch.save(sampler.state_dict(), f"./sampler_aid_epoch_{i}.pth")
            
            # Save best model based on loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(sampler.state_dict(), f"./sampler_aid_best.pth")
                print(f"Saved best model with loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model on AID dataset")
    parser.add_argument("--n_epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use")
    parser.add_argument("--load_pth", type=str, default=None, help="Path to load model from")
    parser.add_argument("--img_size", type=int, default=256, help="Image size to resize to")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--n_feat", type=int, default=256, help="Number of base features in UNet")
    
    args = parser.parse_args()
    
    train_aid(
        n_epoch=args.n_epoch,
        device=args.device,
        load_pth=args.load_pth,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        use_augmentation=not args.no_augmentation,
        n_feat=args.n_feat
    ) 