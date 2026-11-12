"""
SSIM & MS-SSIM Core Mathematical Implementation
Wang et al., IEEE TIP 2004 & IEEE Asilomar 2003
Natively implemented in PyTorch with dynamic data_range calibration,
configurable window sizes, multi-channel handling, and Region-of-Interest (ROI) support.
"""

import math
from typing import Tuple, Optional, Union, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

def _fspecial_gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D normalized Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return gauss / gauss.sum()

def create_gaussian_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    """Creates a 2D Gaussian sliding window for grouped conv2d."""
    _1D_window = _fspecial_gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _auto_detect_data_range(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Infers dynamic intensity range L (1.0 for normalized tensors, 255.0 for 8-bit)."""
    max_val = max(img1.max().item(), img2.max().item())
    return 255.0 if max_val > 1.5 else 1.0

def _ssim_per_channel(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    data_range: float,
    K1: float = 0.01,
    K2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes luminance (l), contrast (c), structure (s), and combined SSIM map.
    Returns: (ssim_map, cs_map, l_map, sigma1_sq)
    """
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    # Numerical stability constants calibrated by data_range L
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Luminance and Contrast-Structure components
    l_map = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = l_map * cs_map

    return ssim_map, cs_map, l_map, sigma1_sq

def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: Optional[float] = None,
    size_average: bool = True,
    return_map: bool = False,
    sigma: float = 1.5,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Functional Standard SSIM Index (Wang et al. 2004).
    Args:
        img1, img2: Tensors of shape (B, C, H, W) or (C, H, W)
        window_size: Sliding Gaussian window size (default 11)
        data_range: Peak-to-peak intensity range L (auto-detected if None)
        size_average: Average over batch and spatial dimensions
        return_map: If True, returns (mean_ssim, ssim_map)
        sigma: Standard deviation of Gaussian window
        roi: Optional bounding box (ymin, xmin, ymax, xmax) for regional evaluation
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    if img1.shape != img2.shape:
        raise ValueError(f"Input tensors must have identical shape, got {img1.shape} and {img2.shape}")

    if data_range is None:
        data_range = _auto_detect_data_range(img1, img2)

    _, channel, H, W = img1.size()
    window = create_gaussian_window(window_size, channel, sigma).to(img1.device, dtype=img1.dtype)

    ssim_map, _, _, _ = _ssim_per_channel(img1, img2, window, window_size, channel, data_range)

    # Regional SSIM (ROI)
    if roi is not None:
        ymin, xmin, ymax, xmax = roi
        ymin = max(0, min(H - 1, int(ymin)))
        xmin = max(0, min(W - 1, int(xmin)))
        ymax = max(ymin + 1, min(H, int(ymax)))
        xmax = max(xmin + 1, min(W, int(xmax)))
        roi_map = ssim_map[:, :, ymin:ymax, xmin:xmax]
        val = roi_map.mean() if size_average else roi_map.mean(dim=[-3, -2, -1])
        if return_map:
            return val, ssim_map
        return val

    if size_average:
        val = ssim_map.mean()
    else:
        val = ssim_map.mean(dim=[-3, -2, -1])

    if return_map:
        return val, ssim_map
    return val

class SSIM(nn.Module):
    """PyTorch nn.Module for differentiable SSIM loss."""
    def __init__(
        self,
        window_size: int = 11,
        data_range: Optional[float] = None,
        size_average: bool = True,
        channel: int = 1,
        sigma: float = 1.5,
    ):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel
        self.sigma = sigma
        self.register_buffer("window", create_gaussian_window(window_size, channel, sigma))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.size(1) != self.channel or self.window.device != img1.device:
            self.channel = img1.size(1)
            self.window = create_gaussian_window(self.window_size, self.channel, self.sigma).to(img1.device, dtype=img1.dtype)
        data_range = self.data_range or _auto_detect_data_range(img1, img2)
        ssim_map, _, _, _ = _ssim_per_channel(img1, img2, self.window, self.window_size, self.channel, data_range)
        return ssim_map.mean() if self.size_average else ssim_map.mean(dim=[-3, -2, -1])

def ms_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: Optional[float] = None,
    size_average: bool = True,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multi-Scale Structural Similarity Index (MS-SSIM) (Wang et al. 2003).
    Evaluates contrast and structure across 5 pyramid scales with 2x2 decimation.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    if data_range is None:
        data_range = _auto_detect_data_range(img1, img2)

    # Standard Wang et al. 2003 empirical scale weights
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device, dtype=img1.dtype)

    levels = weights.shape[0]
    _, channel, H, W = img1.size()

    # Verify resolution supports downsampling
    max_levels = int(math.log2(min(H, W) / window_size)) + 1
    if levels > max_levels:
        weights = weights[:max(1, max_levels)]
        weights = weights / weights.sum()
        levels = len(weights)

    window = create_gaussian_window(window_size, channel).to(img1.device, dtype=img1.dtype)

    mcs = []
    for i in range(levels):
        ssim_map, cs_map, l_map, _ = _ssim_per_channel(img1, img2, window, window_size, channel, data_range)
        if i < levels - 1:
            mcs.append(cs_map.mean(dim=[-3, -2, -1]))
            # 2x2 decimation downsampling
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)
        else:
            # Final scale includes luminance
            mcs.append(ssim_map.mean(dim=[-3, -2, -1]))

    # Weighted geometric mean in log space for numerical stability
    mcs_stack = torch.stack(mcs, dim=-1)
    mcs_stack = torch.clamp(mcs_stack, min=1e-8)
    overall_ms_ssim = torch.exp(torch.sum(weights * torch.log(mcs_stack), dim=-1))

    return overall_ms_ssim.mean() if size_average else overall_ms_ssim

class MS_SSIM(nn.Module):
    """PyTorch nn.Module for Multi-Scale SSIM loss."""
    def __init__(self, window_size: int = 11, data_range: Optional[float] = None, size_average: bool = True):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ms_ssim(img1, img2, window_size=self.window_size, data_range=self.data_range, size_average=self.size_average)

class SSIMLoss(nn.Module):
    """PyTorch loss function computing 1.0 - SSIM for gradient-based optimization."""
    def __init__(
        self,
        window_size: int = 11,
        data_range: Optional[float] = None,
        size_average: bool = True,
        channel: int = 1,
        sigma: float = 1.5,
    ):
        super(SSIMLoss, self).__init__()
        self.ssim_module = SSIM(
            window_size=window_size,
            data_range=data_range,
            size_average=size_average,
            channel=channel,
            sigma=sigma,
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.ssim_module(img1, img2)

