"""
Image Restoration & Enhancement Lab
Classical filtering, edge-preserving denoising, and sharpening with metric recovery tracking (Delta SSIM).
"""

from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from .ssim import ssim
from .metrics import psnr
from .distortions import apply_gaussian_blur

def gaussian_filter(tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Low-pass noise reduction filter."""
    k = max(3, int(round(sigma * 4)) | 1)
    return apply_gaussian_blur(tensor, kernel_size=k, sigma=sigma)

def median_filter(tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Non-linear median filtering for impulsive salt-and-pepper noise."""
    if kernel_size <= 1:
        return tensor.clone()
    pad = kernel_size // 2
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="replicate")
    # Unfold spatial patches (B, C * K * K, H * W)
    patches = F.unfold(padded, kernel_size=kernel_size)
    B, C, H, W = tensor.shape
    patches = patches.view(B, C, kernel_size * kernel_size, H * W)
    # Median along patch dimension
    median_vals, _ = torch.median(patches, dim=2)
    return median_vals.view(B, C, H, W)

def bilateral_filter(tensor: torch.Tensor, kernel_size: int = 5, sigma_s: float = 3.0, sigma_r: float = 0.1) -> torch.Tensor:
    """Edge-preserving bilateral smoothing."""
    pad = kernel_size // 2
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="replicate")
    B, C, H, W = tensor.shape

    # Spatial weights
    coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - pad
    dist_sq = coords.view(-1, 1) ** 2 + coords.view(1, -1) ** 2
    spatial_w = torch.exp(-dist_sq / (2.0 * sigma_s ** 2)).view(1, 1, kernel_size * kernel_size, 1)

    patches = F.unfold(padded, kernel_size=kernel_size).view(B, C, kernel_size * kernel_size, H * W)
    center = tensor.view(B, C, 1, H * W)

    # Range / photometric intensity weights
    range_diff = patches - center
    range_w = torch.exp(-(range_diff ** 2) / (2.0 * sigma_r ** 2))

    total_w = spatial_w * range_w
    norm_w = total_w.sum(dim=2, keepdim=True) + 1e-8
    filtered = (patches * total_w).sum(dim=2, keepdim=True) / norm_w
    return filtered.view(B, C, H, W)

def unsharp_mask(tensor: torch.Tensor, sigma: float = 1.5, amount: float = 1.0) -> torch.Tensor:
    """High-frequency unsharp masking / edge sharpening."""
    blurred = apply_gaussian_blur(tensor, kernel_size=9, sigma=sigma)
    sharpened = tensor + amount * (tensor - blurred)
    return torch.clamp(sharpened, 0.0, 1.0)

def benchmark_restorations(ref: torch.Tensor, distorted: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Executes all restoration algorithms on the distorted image,
    calculates SSIM & PSNR, and measures the quality recovery delta (Delta SSIM).
    """
    base_ssim = ssim(ref, distorted).item()
    base_psnr = psnr(ref, distorted)

    results = [
        {
            "technique": "Baseline (Unrestored Distorted)",
            "ssim": round(base_ssim, 4),
            "psnr_db": round(base_psnr, 2),
            "delta_ssim": 0.0,
            "status": "Baseline",
        }
    ]

    techniques = [
        ("Median Filter (3x3)", median_filter(distorted, kernel_size=3)),
        ("Gaussian Filter (σ=1.2)", gaussian_filter(distorted, sigma=1.2)),
        ("Bilateral Filter (Edge-Preserve)", bilateral_filter(distorted, kernel_size=5, sigma_s=2.5, sigma_r=0.15)),
        ("Unsharp Mask (Sharpening)", unsharp_mask(distorted, sigma=1.0, amount=0.8)),
    ]

    for name, restored in techniques:
        score_ssim = ssim(ref, restored).item()
        score_psnr = psnr(ref, restored)
        delta = score_ssim - base_ssim

        results.append({
            "technique": name,
            "ssim": round(score_ssim, 4),
            "psnr_db": round(score_psnr, 2),
            "delta_ssim": round(delta, 4),
            "status": "Improved" if delta > 0.005 else ("Degraded" if delta < -0.005 else "Neutral"),
        })

    return results
