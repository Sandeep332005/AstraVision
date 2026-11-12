"""
Parametric Distortion Laboratory
Simulates real-world optical, sensor, compression, and transmission degradation:
Gaussian Noise, Salt-and-Pepper, Gaussian Blur, Motion Blur, Genuine JPEG Compression, Contrast/Brightness, and Resizing.
"""

from typing import Optional
import io
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def add_gaussian_noise(tensor: torch.Tensor, sigma: float = 0.1, seed: Optional[int] = None) -> torch.Tensor:
    """Adds zero-mean Gaussian noise with standard deviation sigma and optional seed."""
    if sigma <= 0:
        return tensor.clone()
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)

def add_salt_and_pepper_noise(tensor: torch.Tensor, prob: float = 0.02, seed: Optional[int] = None) -> torch.Tensor:
    """Adds impulsive salt (1.0) and pepper (0.0) noise with optional seed."""
    if prob <= 0:
        return tensor.clone()
    if seed is not None:
        torch.manual_seed(seed)
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

def apply_motion_blur(tensor: torch.Tensor, size: int = 15, angle: float = 45.0) -> torch.Tensor:
    """Applies linear directional motion blur along specified angle in degrees."""
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

def apply_real_jpeg_compression(tensor: torch.Tensor, quality: int = 15) -> torch.Tensor:
    """
    Applies genuine JPEG Discrete Cosine Transform (DCT) quantization using standard JPEG encoder.
    quality: JPEG Quality factor [1, 95] (lower = more severe blocking and compression artifacts).
    """
    quality = max(1, min(95, int(quality)))
    if tensor.dim() == 4:
        squeeze_batch = True
        img_t = tensor.squeeze(0)
    else:
        squeeze_batch = False
        img_t = tensor

    # Convert to PIL Image
    np_img = (img_t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)

    # Compress into in-memory JPEG buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Read back compressed image
    decompressed = Image.open(buffer).convert("RGB")
    np_decomp = np.array(decompressed, dtype=np.float32) / 255.0
    out_tensor = torch.from_numpy(np_decomp).permute(2, 0, 1).to(tensor.device)

    if squeeze_batch:
        out_tensor = out_tensor.unsqueeze(0)
    return out_tensor

def apply_resizing_distortion(tensor: torch.Tensor, scale_factor: float = 0.25) -> torch.Tensor:
    """Downsamples and upsamples image to evaluate decimation & interpolation artifacts."""
    if scale_factor >= 1.0:
        return tensor.clone()
    _, _, H, W = tensor.shape
    small_h = max(4, int(H * scale_factor))
    small_w = max(4, int(W * scale_factor))

    down = F.interpolate(tensor, size=(small_h, small_w), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
    return up

def adjust_brightness_contrast(tensor: torch.Tensor, brightness: float = 0.0, contrast: float = 1.0) -> torch.Tensor:
    """Parametric brightness offset and contrast scaling."""
    out = (tensor - 0.5) * contrast + 0.5 + brightness
    return torch.clamp(out, 0.0, 1.0)
