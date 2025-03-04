from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os
from PIL import Image

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

    # Create output directory if it doesn't exist
    os.makedirs("./contents/aid", exist_ok=True)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        sampler.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = sampler(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

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


if __name__ == "__main__":
    train_aid() 