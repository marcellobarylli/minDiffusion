from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddim import DDIM

# set gpu to 2
torch.cuda.set_device(2)


def train_cifar10(
    n_epoch: int = 100, device: str = "cuda:2", load_pth: Optional[str] = None
) -> None:

    sampler = DDIM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), eta=0.5, n_T=1000)

    if load_pth is not None:
        sampler.load_state_dict(torch.load("sampler_cifar.pth"))

    sampler.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(sampler.parameters(), lr=1e-5)

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
            xh = sampler.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/sampler_sample_cifar{i}.png")

            # save model
            torch.save(sampler.state_dict(), f"./sampler_cifar.pth")


if __name__ == "__main__":
    train_cifar10()
