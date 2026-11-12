"""
Parametric Distortion Laboratory
Simulates real-world optical, transmission, and sensor degradation:
Gaussian Noise, Impulse Salt-and-Pepper, Gaussian Blur, Motion Blur, JPEG Artifacts, and Luminance Shift.
"""

import math
import torch
import torch.nn.functional as F

def add_gaussian_noise(tensor: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Adds zero-mean Gaussian noise with standard deviation sigma."""
    if sigma <= 0:
        return tensor.clone()
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)

def add_salt_and_pepper_noise(tensor: torch.Tensor, prob: float = 0.02) -> torch.Tensor:
    """Adds impulsive salt (1.0) and pepper (0.0) noise."""
    if prob <= 0:
        return tensor.clone()
    out = tensor.clone()
    rand = torch.rand_like(tensor)
    out[rand < (prob / 2.0)] = 0.0
    out[(rand >= (prob / 2.0)) & (rand < prob)] = 1.0
    return out

def apply_gaussian_blur(tensor: torch.Tensor, kernel_size: int = 15, sigma: float = 2.5) -> torch.Tensor:
    """Applies Gaussian spatial low-pass filtering."""
    if sigma <= 0.1:
        return tensor.clone()
    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - kernel_size // 2
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(1).mm(g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    channels = tensor.size(1)
    kernel = g2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    return F.conv2d(tensor, kernel, padding=pad, groups=channels)

def apply_motion_blur(tensor: torch.Tensor, size: int = 15, angle: float = 0.0) -> torch.Tensor:
    """Applies linear directional motion blur."""
    if size <= 1:
        return tensor.clone()
    kernel = torch.zeros((size, size), device=tensor.device, dtype=torch.float32)
    center = size // 2
    rad = math.radians(angle)
    dx = math.cos(rad)
    dy = math.sin(rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    else:
        kernel = kernel / kernel.sum()

    channels = tensor.size(1)
    kernel2d = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()
    pad = size // 2
    return F.conv2d(tensor, kernel2d, padding=pad, groups=channels)

def apply_jpeg_artifacts(tensor: torch.Tensor, block_size: int = 8, quant_step: float = 0.15) -> torch.Tensor:
    """Simulates 8x8 block-based DCT compression quantization artifacts."""
    if block_size <= 1:
        return tensor.clone()
    B, C, H, W = tensor.shape
    # Pad to multiple of block_size
    pad_h = (block_size - (H % block_size)) % block_size
    pad_w = (block_size - (W % block_size)) % block_size
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    # Downsample average and quantize
    down = F.avg_pool2d(padded, kernel_size=block_size, stride=block_size)
    if quant_step > 0:
        down = torch.round(down / quant_step) * quant_step

    # Upsample block structure
    up = F.interpolate(down, size=(padded.shape[-2], padded.shape[-1]), mode="nearest")
    # Blend blocky edges
    blended = 0.4 * padded + 0.6 * up
    return torch.clamp(blended[:, :, :H, :W], 0.0, 1.0)

def adjust_brightness_contrast(tensor: torch.Tensor, brightness: float = 0.0, contrast: float = 1.0) -> torch.Tensor:
    """Parametric brightness offset and contrast scaling."""
    out = (tensor - 0.5) * contrast + 0.5 + brightness
    return torch.clamp(out, 0.0, 1.0)
