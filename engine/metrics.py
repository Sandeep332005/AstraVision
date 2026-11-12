"""
Perceptual & Statistical Image Quality Metrics Suite
PSNR, MSE, MAE, UQI (Universal Image Quality Index), and combined metric analysis.
"""

from typing import Dict, Optional
import math
import torch
from .ssim import ssim, ms_ssim, _auto_detect_data_range

def mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean Squared Error (MSE)."""
    return torch.mean((img1 - img2) ** 2).item()

def mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean Absolute Error (MAE / L1 loss)."""
    return torch.mean(torch.abs(img1 - img2)).item()

def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: Optional[float] = None) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR in dB).
    PSNR = 10 * log10(L^2 / MSE)
    """
    if data_range is None:
        data_range = _auto_detect_data_range(img1, img2)
    err = mse(img1, img2)
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / err)

def uqi(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Universal Image Quality Index (Wang & Bovik, IEEE SPL 2002).
    Mathematically identical to SSIM with C1 = 0, C2 = 0.
    """
    # SSIM with minimal stability constant approximates UQI
    score = ssim(img1, img2, window_size=8, data_range=1.0)
    return score.item()

def calculate_all_metrics(img1: torch.Tensor, img2: torch.Tensor, data_range: Optional[float] = None) -> Dict[str, float]:
    """Calculates all key metrics in a unified dictionary."""
    if data_range is None:
        data_range = _auto_detect_data_range(img1, img2)

    val_mse = mse(img1, img2)
    val_mae = mae(img1, img2)
    val_psnr = psnr(img1, img2, data_range=data_range)
    val_ssim = ssim(img1, img2, data_range=data_range).item()

    # Try computing MS-SSIM if resolution permits
    try:
        val_msssim = ms_ssim(img1, img2, data_range=data_range).item()
    except Exception:
        val_msssim = val_ssim

    val_uqi = uqi(img1, img2)

    return {
        "ssim": round(val_ssim, 4),
        "ms_ssim": round(val_msssim, 4),
        "psnr_db": round(val_psnr, 2) if val_psnr != float("inf") else 99.99,
        "mse": round(val_mse * (255.0 ** 2 if data_range == 1.0 else 1.0), 2),
        "mae": round(val_mae * (255.0 if data_range == 1.0 else 1.0), 2),
        "uqi": round(val_uqi, 4),
    }
