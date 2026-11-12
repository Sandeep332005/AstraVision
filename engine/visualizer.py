"""
Visual Diagnostic & Heatmap Generation Engine
Generates 2D local SSIM quality maps, absolute error maps, edge distortion maps,
adjustable threshold masks, and alpha-blended overlays.
"""

from typing import Tuple, Dict, Any, Optional
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
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    else:
        gray = tensor

    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=tensor.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=tensor.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return mag / (mag.max() + 1e-8)

def compute_diagnostic_maps(
    ref: torch.Tensor,
    dist: torch.Tensor,
    threshold: float = 0.80,
    overlay_alpha: float = 0.50,
) -> Dict[str, np.ndarray]:
    """
    Computes diagnostic heatmaps as 2D numpy arrays normalized [0.0, 1.0]:
    - ssim_map: local structural similarity
    - error_map: absolute pixel difference |I1 - I2|
    - edge_diff_map: structural boundary discrepancy
    - threshold_mask: binary mask of severe degradation (SSIM < threshold)
    - overlay_map: alpha-blended overlay of heatmap over reference image
    """
    _, ssim_map_t = ssim(ref, dist, return_map=True)
    if ssim_map_t.dim() == 4 and ssim_map_t.size(1) > 1:
        ssim_map_t = ssim_map_t.mean(dim=1, keepdim=True)

    diff = torch.abs(ref - dist)
    if diff.size(1) > 1:
        error_map_t = diff.mean(dim=1, keepdim=True)
    else:
        error_map_t = diff

    # Edge discrepancy
    edges_ref = _sobel_edges(ref)
    edges_dist = _sobel_edges(dist)
    edge_diff_t = torch.abs(edges_ref - edges_dist)

    ssim_arr = np.clip(ssim_map_t.squeeze().detach().cpu().numpy(), 0.0, 1.0)
    error_arr = np.clip(error_map_t.squeeze().detach().cpu().numpy(), 0.0, 1.0)
    edge_arr = np.clip(edge_diff_t.squeeze().detach().cpu().numpy(), 0.0, 1.0)

    # Threshold mask: pixels with severe degradation
    threshold_mask = (ssim_arr < threshold).astype(np.float32)

    return {
        "ssim_map": ssim_arr,
        "error_map": error_arr,
        "edge_diff_map": edge_arr,
        "threshold_mask": threshold_mask,
    }

def array_to_base64_image(arr: np.ndarray, colormap: str = "jet", ref_bg: Optional[np.ndarray] = None, alpha: float = 0.6) -> str:
    """Converts a 2D float array [0, 1] into a base64 PNG data URL with optional background overlay."""
    arr = np.clip(arr, 0.0, 1.0)

    if colormap == "jet":
        r = np.clip(1.5 - np.abs(arr * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(arr * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(arr * 4 - 1), 0, 1)
        heat_rgb = np.stack([r, g, b], axis=-1)
    elif colormap == "hot":
        r = np.clip(arr * 3.0, 0, 1)
        g = np.clip(arr * 3.0 - 1.0, 0, 1)
        b = np.clip(arr * 3.0 - 2.0, 0, 1)
        heat_rgb = np.stack([r, g, b], axis=-1)
    elif colormap == "mask":
        # Crimson mask for threshold violations
        r = arr * 0.95
        g = arr * 0.20
        b = arr * 0.25
        heat_rgb = np.stack([r, g, b], axis=-1)
    else:
        heat_rgb = np.stack([arr, arr, arr], axis=-1)

    if ref_bg is not None:
        if ref_bg.ndim == 2:
            ref_rgb = np.stack([ref_bg, ref_bg, ref_bg], axis=-1)
        else:
            ref_rgb = ref_bg
        blended = (1.0 - alpha) * ref_rgb + alpha * heat_rgb
        uint8_img = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    else:
        uint8_img = (heat_rgb * 255).astype(np.uint8)

    pil_img = Image.fromarray(uint8_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
