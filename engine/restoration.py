"""
Image Restoration & Enhancement Lab
Classical filtering, edge-preserving denoising, Non-Local Means (NLM), and unsharp masking.
Measures execution time, memory usage, and quality recovery delta (Delta SSIM & Delta PSNR).
"""

import time
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from .ssim import ssim, ms_ssim
from .metrics import psnr, mse
from .distortions import apply_gaussian_blur

def gaussian_filter(tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Low-pass Gaussian noise reduction filter."""
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

def bilateral_filter(tensor: torch.Tensor, kernel_size: int = 5, sigma_s: float = 3.0, sigma_r: float = 0.15) -> torch.Tensor:
    """Edge-preserving bilateral smoothing (photometric range + spatial domain)."""
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

def non_local_means_denoising(tensor: torch.Tensor, h: float = 0.1, patch_size: int = 3, search_window: int = 9) -> torch.Tensor:
    """
    Non-Local Means (NLM) patch denoising (Buades et al. 2005).
    Replaces each pixel with weighted average of similar patches across the search window.
    """
    B, C, H, W = tensor.shape
    pad_s = search_window // 2
    padded = F.pad(tensor, (pad_s, pad_s, pad_s, pad_s), mode="replicate")

    # Fast sliding window search
    weights_sum = torch.zeros((B, 1, H, W), device=tensor.device)
    output = torch.zeros_like(tensor)
    h2 = h * h

    for dy in range(-pad_s, pad_s + 1, 2):
        for dx in range(-pad_s, pad_s + 1, 2):
            shifted = padded[:, :, pad_s + dy : pad_s + dy + H, pad_s + dx : pad_s + dx + W]
            diff_sq = (tensor - shifted) ** 2
            # Patch similarity approximated via local blur
            dist = apply_gaussian_blur(diff_sq, kernel_size=patch_size, sigma=1.0)
            w = torch.exp(-dist.mean(dim=1, keepdim=True) / (h2 + 1e-8))
            output += shifted * w
            weights_sum += w

    return output / (weights_sum + 1e-8)

def unsharp_mask(tensor: torch.Tensor, sigma: float = 1.5, amount: float = 1.0) -> torch.Tensor:
    """High-frequency unsharp masking / edge sharpening."""
    blurred = apply_gaussian_blur(tensor, kernel_size=9, sigma=sigma)
    sharpened = tensor + amount * (tensor - blurred)
    return torch.clamp(sharpened, 0.0, 1.0)

def benchmark_restorations(ref: torch.Tensor, distorted: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Executes the full suite of restoration algorithms on the distorted image,
    benchmarks execution runtime in milliseconds, computes SSIM & PSNR,
    and calculates metric recovery deltas (Delta SSIM).
    """
    t0 = time.perf_counter_ns()
    base_ssim = ssim(ref, distorted).item()
    base_msssim = ms_ssim(ref, distorted).item()
    base_psnr = psnr(ref, distorted)
    base_mse = mse(ref, distorted)
    t_base = (time.perf_counter_ns() - t0) / 1e6

    mem_bytes = distorted.element_size() * distorted.nelement()
    mem_kb = round(mem_bytes / 1024.0, 1)

    results = [
        {
            "algorithm": "Baseline (Unrestored Distorted)",
            "technique": "Baseline (Unrestored Distorted)",
            "ssim": round(base_ssim, 4),
            "ms_ssim": round(base_msssim, 4),
            "psnr_db": round(base_psnr, 2),
            "mse": round(base_mse * (255.0 ** 2), 2),
            "delta_ssim": 0.0,
            "runtime_ms": round(t_base, 2),
            "memory_kb": mem_kb,
            "status": "Baseline",
        }
    ]

    algorithms = [
        ("Median Filter (3x3)", lambda t: median_filter(t, kernel_size=3)),
        ("Gaussian Filter (σ=1.2)", lambda t: gaussian_filter(t, sigma=1.2)),
        ("Bilateral Filter (Edge-Preserve)", lambda t: bilateral_filter(t, kernel_size=5, sigma_s=2.5, sigma_r=0.15)),
        ("Non-Local Means (NLM Denoise)", lambda t: non_local_means_denoising(t, h=0.08)),
        ("Unsharp Mask (Sharpening)", lambda t: unsharp_mask(t, sigma=1.0, amount=0.8)),
    ]

    for name, fn in algorithms:
        start_t = time.perf_counter_ns()
        restored = fn(distorted)
        runtime_ms = (time.perf_counter_ns() - start_t) / 1e6

        score_ssim = ssim(ref, restored).item()
        score_msssim = ms_ssim(ref, restored).item()
        score_psnr = psnr(ref, restored)
        score_mse = mse(ref, restored)
        delta = score_ssim - base_ssim

        status = "Improved" if delta > 0.005 else ("Degraded" if delta < -0.005 else "Neutral")

        results.append({
            "algorithm": name,
            "technique": name,
            "ssim": round(score_ssim, 4),
            "ms_ssim": round(score_msssim, 4),
            "psnr_db": round(score_psnr, 2),
            "mse": round(score_mse * (255.0 ** 2), 2),
            "delta_ssim": round(delta, 4),
            "runtime_ms": round(runtime_ms, 2),
            "memory_kb": mem_kb,
            "status": status,
        })

    return results
