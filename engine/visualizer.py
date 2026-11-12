"""
Visual Diagnostic & Heatmap Generation Engine
Generates 2D local SSIM quality maps, absolute error maps, edge distortion maps, and colormapping.
"""

from typing import Tuple, Dict, Any
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from .ssim import ssim

def _sobel_edges(tensor: torch.Tensor) -> torch.Tensor:
    """Computes spatial gradient magnitude using Sobel operators."""
    if tensor.size(1) > 1:
        # Grayscale luminance weights
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    else:
        gray = tensor

    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=tensor.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=tensor.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return mag / (mag.max() + 1e-8)

def compute_diagnostic_maps(ref: torch.Tensor, dist: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Computes diagnostic heatmaps as 2D numpy arrays normalized [0.0, 1.0]:
    - local_ssim_map: local structural similarity
    - error_map: absolute pixel difference |I1 - I2|
    - edge_diff_map: structural boundary discrepancy
    """
    _, ssim_map = ssim(ref, dist, return_map=True)
    if ssim_map.dim() == 4 and ssim_map.size(1) > 1:
        ssim_map = ssim_map.mean(dim=1, keepdim=True)

    diff = torch.abs(ref - dist)
    if diff.size(1) > 1:
        error_map = diff.mean(dim=1, keepdim=True)
    else:
        error_map = diff

    # Edge discrepancy
    edges_ref = _sobel_edges(ref)
    edges_dist = _sobel_edges(dist)
    edge_diff = torch.abs(edges_ref - edges_dist)

    return {
        "ssim_map": np.clip(ssim_map.squeeze().detach().cpu().numpy(), 0.0, 1.0),
        "error_map": np.clip(error_map.squeeze().detach().cpu().numpy(), 0.0, 1.0),
        "edge_diff_map": np.clip(edge_diff.squeeze().detach().cpu().numpy(), 0.0, 1.0),
    }

def array_to_base64_image(arr: np.ndarray, colormap: str = "jet") -> str:
    """Converts a 2D float array [0, 1] into a base64 PNG data URL."""
    H, W = arr.shape
    arr = np.clip(arr, 0.0, 1.0)

    if colormap == "jet":
        # Approximate perceptual jet colormap
        r = np.clip(1.5 - np.abs(arr * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(arr * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(arr * 4 - 1), 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
    elif colormap == "hot":
        r = np.clip(arr * 3.0, 0, 1)
        g = np.clip(arr * 3.0 - 1.0, 0, 1)
        b = np.clip(arr * 3.0 - 2.0, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
    else:
        # Grayscale
        rgb = np.stack([arr, arr, arr], axis=-1)

    uint8_img = (rgb * 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
