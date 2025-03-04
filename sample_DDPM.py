import torch
from torchvision.utils import save_image, make_grid
from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

# Use the same device as training
device = "cuda:2"

# Create model with the same parameters as in training
# Note: We're using DDPM instead of DDIM here
model = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

# Load the saved model weights
# Even though the model was trained as DDIM, the core network (eps_model) is the same
# so we can load the weights into a DDPM model
model.load_state_dict(torch.load("ddpm_cifar.pth"), strict=False)
model.to(device)
model.eval()

num_samples = 8

print(f"Sampling {num_samples} images using DDPM sampler...")

# Sample images
with torch.no_grad():
    # Generate 64 images with size 3x32x32 (CIFAR-10 dimensions)
    samples = model.sample(num_samples, (3, 32, 32), device)
    
    # Create and save a grid of images
    grid = make_grid(samples, normalize=True, value_range=(-1, 1), nrow=num_samples)
    save_image(grid, "./contents/sample_ddpm_64.png")
    
    # Also save individual images for closer inspection
    for i in range(num_samples):
        save_image(samples[i], f"./contents/sample_ddpm_{i}.png", normalize=True, value_range=(-1, 1))

print(f"DDPM sampling completed. Check ./contents/sample_ddpm_{num_samples}.png and individual images in ./contents/")