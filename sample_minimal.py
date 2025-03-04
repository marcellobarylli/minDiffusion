import torch
from torchvision.utils import save_image, make_grid
from mindiffusion.unet import NaiveUnet
from mindiffusion.ddim import DDIM

# Use the same device as training
device = "cuda:2"

# Create model with the same parameters as in training
model = DDIM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), eta=0, n_T=1000)

# Load the saved model weights
model.load_state_dict(torch.load("ddpm_cifar.pth"))
model.to(device)
model.eval()

# Sample images
with torch.no_grad():
    # Generate 64 images with size 3x32x32 (CIFAR-10 dimensions)
    samples = model.sample(64, (3, 32, 32), device)
    
    # Create and save a grid of images
    grid = make_grid(samples, normalize=True, value_range=(-1, 1), nrow=8)
    save_image(grid, "./contents/sample_minimal_64.png")
    
    # Also save individual images for closer inspection
    for i in range(64):
        save_image(samples[i], f"./contents/sample_{i}.png", normalize=True, value_range=(-1, 1))

print("Sampling completed. Check ./contents/sample_minimal_64.png and individual images in ./contents/") 