import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
import random

import numpy as np
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
from tqdm import tqdm

from float8_to_bits import float8_vec_to_bits, bit_vec_to_float8
from stable_diffusion import ddrm_posterior_sampling as posterior_sampler, vae_encode, vae_decode
from stable_diffusion import nested_posterior_sampling as restoration_function


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def initialize_stable():
    """
    initialize the stable diffusion model for the following execution to be deterministic
    """
    device = torch.device("cuda")
    c, h, w = 4, 64, 64
    noise = torch.randn((2, c, h, w), device=device)
    H = torch.zeros((0, c * h * w), dtype=torch.float32, device=device)  # Empty sensing matrix
    y = torch.zeros((0, 1), dtype=torch.float32, device=device) # Empty measurements
    _ = posterior_sampler(noise, H, y, prompt="").to(torch.float32)

initialize_stable()

def entropy_encode(bytes):
    return bytes

def entropy_decode(bytes):
    return bytes

@torch.no_grad()
def SelectNewRows(H, y, r, shape, s=None, **sampling_kwargs):
    c, h, w = shape
    s = s if s is not None else (r * 4) // 3
    noise = torch.randn((s, c, h, w), device=torch.device("cuda"))
    samples = posterior_sampler(noise, H, y, **sampling_kwargs).to(torch.float32)
    samples = samples.reshape(s, -1)
    samples = samples - samples.mean(0, keepdim=True)
    new_rows = torch.linalg.svd(samples, full_matrices=False)[-1][:r]
    return new_rows

@torch.no_grad()
def PSC_compress(image, N, r, num_samples, **sampling_kwargs):
    c, h, w = image.shape
    device = image.device
    H = torch.zeros((0, c * h * w), dtype=torch.float32, device=device) # Empty sensing matrix
    y = H @ image.reshape((-1, 1)) # Empty measurements

    for n in tqdm(range(N), leave=False, desc="Compressing"):
        new_rows = SelectNewRows(H, y, r, (c, h, w), num_samples, **sampling_kwargs)
        H = torch.cat([H, new_rows])
        y = torch.cat([y, new_rows @ image.reshape((-1, 1))])

        y = y.to(torch.float8_e4m3fn).to(torch.float32) # Quantize

    compressed_representation = float8_vec_to_bits(y.cpu())
    return entropy_encode(compressed_representation)

@torch.no_grad()
def PSC_decompress(compressed_representation, N, r, num_samples, shape, num_restorations=1, **sampling_kwargs):
    compressed_representation = entropy_decode(compressed_representation)
    compressed_representation = bit_vec_to_float8(compressed_representation).to(torch.float32).to(torch.device("cuda"))
    c, h, w = shape
    device = compressed_representation.device
    H = torch.zeros((0, c * h * w), dtype=torch.float32, device=device) # Empty sensing matrix
    y = torch.zeros((0, 1), dtype=torch.float32, device=device) # Empty measurements

    for n in tqdm(range(N), leave=False, desc="Decompressing"):
        new_rows = SelectNewRows(H, y, r, (c, h, w), num_samples, **sampling_kwargs)
        H = torch.cat([H, new_rows])
        y = compressed_representation[:(n*r + r)]

    noise = torch.randn((num_restorations, c, h, w), device=device)
    return restoration_function(noise, H, y, **sampling_kwargs)

@torch.no_grad()
def PSC_Simulate(image, N, r, num_samples, num_restorations=1, **sampling_kwargs):
    c, h, w = image.shape
    device = image.device
    H = torch.zeros((0, c * h * w), dtype=torch.float32, device=device) # Empty sensing matrix
    y = H @ image.reshape((-1, 1)) # Empty measurements

    for n in tqdm(range(N), leave=False, desc="Compressing"):
        new_rows = SelectNewRows(H, y, r, (c, h, w), num_samples, **sampling_kwargs)
        H = torch.cat([H, new_rows])
        y = torch.cat([y, new_rows @ image.reshape((-1, 1))])

        y = y.to(torch.float8_e4m3fn).to(torch.float32) # Quantize

    compressed_representation = float8_vec_to_bits(y.cpu())
    noise = torch.randn((num_restorations, c, h, w), device=device)
    return entropy_encode(compressed_representation), restoration_function(noise, H, y, **sampling_kwargs)

@torch.no_grad()
def Latent_PSC_compress(image, prompt, N, r, num_samples):
    latent = vae_encode(image)[0]
    return PSC_compress(latent, N, r, num_samples, prompt=prompt)

@torch.no_grad()
def Latent_PSC_decompress(compressed_representation, prompt, N, r, num_samples, shape=(4, 64, 64), num_restorations=1):
    latents = PSC_decompress(compressed_representation, N, r, num_samples, shape, num_restorations=num_restorations, prompt=prompt)
    return vae_decode(latents)[0]

@torch.no_grad()
def Latent_PSC_simulate(image, prompt, N, r, num_samples, num_restorations=1):
    latent = vae_encode(image)[0]
    byte_stream, latents = PSC_Simulate(latent, N, r, num_samples, num_restorations=1, prompt=prompt)
    return byte_stream, vae_decode(latents)[0]
