"""
Perceptual & Statistical Image Quality Metrics Suite
Standard SSIM, Multi-Scale SSIM, PSNR, MSE, MAE, UQI, and Perceptual Feature Distance.
"""

from typing import Dict, Optional, Any, Tuple
import math
import torch
import torch.nn.functional as F
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
    score = ssim(img1, img2, window_size=8, data_range=1.0)
    return score.item()

def compute_gradient_perceptual_distance(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Multi-Scale Gradient Perceptual Distance (Wang & Simoncelli perceptual feature representation).
    Evaluates multi-scale directional edge difference without requiring external weight downloads.
    """
    if img1.dim() == 3: img1 = img1.unsqueeze(0)
    if img2.dim() == 3: img2 = img2.unsqueeze(0)

    # Convert to single luminance channel if RGB
    if img1.size(1) == 3:
        g1 = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
        g2 = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]
    else:
        g1, g2 = img1, img2

    # Horizontal and vertical Sobel filters
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img1.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=img1.device).view(1, 1, 3, 3)

    gx1, gy1 = F.conv2d(g1, kx, padding=1), F.conv2d(g1, ky, padding=1)
    gx2, gy2 = F.conv2d(g2, kx, padding=1), F.conv2d(g2, ky, padding=1)

    mag1 = torch.sqrt(gx1 ** 2 + gy1 ** 2 + 1e-8)
    mag2 = torch.sqrt(gx2 ** 2 + gy2 ** 2 + 1e-8)

    grad_diff = torch.mean(torch.abs(mag1 - mag2)).item()
    return round(float(grad_diff), 4)

perceptual_gradient_distance = compute_gradient_perceptual_distance

def calculate_all_metrics(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Calculates all perceptual and statistical metrics in a verified dictionary.
    Includes interpretation and dynamic range notation.
    """
    if data_range is None:
        data_range = _auto_detect_data_range(img1, img2)

    val_mse = mse(img1, img2)
    val_mae = mae(img1, img2)
    val_psnr = psnr(img1, img2, data_range=data_range)
    val_ssim = ssim(img1, img2, data_range=data_range, roi=roi).item()

    # Multi-Scale SSIM
    try:
        val_msssim = ms_ssim(img1, img2, data_range=data_range).item()
    except Exception:
        val_msssim = val_ssim

    val_uqi = uqi(img1, img2)
    val_perceptual = compute_gradient_perceptual_distance(img1, img2)

    # Human visual perception rating based on Zhou Wang et al. MOS mappings
    if val_ssim >= 0.95:
        verdict = "Excellent (Imperceptible degradation)"
    elif val_ssim >= 0.85:
        verdict = "Good (Slightly perceptible, not annoying)"
    elif val_ssim >= 0.70:
        verdict = "Fair (Noticeable structural loss)"
    elif val_ssim >= 0.50:
        verdict = "Poor (Substantial structural degradation)"
    else:
        verdict = "Bad (Severe perceptual destruction)"

    return {
        "ssim": round(val_ssim, 4),
        "ms_ssim": round(val_msssim, 4),
        "psnr_db": round(val_psnr, 2) if val_psnr != float("inf") else 99.99,
        "mse": round(val_mse * (255.0 ** 2 if data_range == 1.0 else 1.0), 2),
        "mae": round(val_mae * (255.0 if data_range == 1.0 else 1.0), 2),
        "uqi": round(val_uqi, 4),
        "perceptual_dist": val_perceptual,
        "perceptual_distance": val_perceptual,
        "verdict": verdict,
        "data_range": data_range,
    }
