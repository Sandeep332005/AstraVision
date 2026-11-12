"""
SSIM LAB Formal Verification & Mathematical Unit Tests
Validates identity, symmetry, dynamic range invariance, and degradation benchmarks.
"""

import math
import torch
import pytest
from engine.ssim import ssim, ms_ssim, SSIM, MS_SSIM
from engine.metrics import calculate_all_metrics, psnr, mse
from engine.distortions import add_gaussian_noise, apply_gaussian_blur
from engine.restoration import median_filter, benchmark_restorations

def test_ssim_identity():
    """Property: SSIM(x, x) == 1.0 (Identical inputs yield perfect fidelity)."""
    torch.manual_seed(42)
    img = torch.rand(1, 1, 128, 128)
    score = ssim(img, img)
    assert abs(score.item() - 1.0) < 1e-4

def test_ssim_symmetry():
    """Property: SSIM(x, y) == SSIM(y, x) (Symmetric evaluation)."""
    torch.manual_seed(42)
    img1 = torch.rand(1, 1, 128, 128)
    img2 = torch.rand(1, 1, 128, 128)
    score1 = ssim(img1, img2)
    score2 = ssim(img2, img1)
    assert abs(score1.item() - score2.item()) < 1e-5

def test_data_range_invariance():
    """Property: SSIM on [0, 1] with L=1.0 must match SSIM on [0, 255] with L=255.0."""
    torch.manual_seed(42)
    img1_float = torch.rand(1, 1, 128, 128)
    img2_float = torch.rand(1, 1, 128, 128)

    score_float = ssim(img1_float, img2_float, data_range=1.0)

    img1_255 = img1_float * 255.0
    img2_255 = img2_float * 255.0
    score_255 = ssim(img1_255, img2_255, data_range=255.0)

    assert abs(score_float.item() - score_255.item()) < 1e-4

def test_ms_ssim_identity():
    """Property: MS-SSIM(x, x) == 1.0 across 5 scales."""
    torch.manual_seed(42)
    img = torch.rand(1, 1, 160, 160)
    score = ms_ssim(img, img)
    assert abs(score.item() - 1.0) < 1e-3

def test_degradation_ordering():
    """Property: SSIM must drop strictly as noise intensity increases."""
    torch.manual_seed(42)
    ref = torch.rand(1, 1, 128, 128)
    mild_noise = add_gaussian_noise(ref, sigma=0.05)
    severe_noise = add_gaussian_noise(ref, sigma=0.30)

    score_mild = ssim(ref, mild_noise).item()
    score_severe = ssim(ref, severe_noise).item()

    assert score_mild > score_severe
    assert score_mild < 1.0

def test_psnr_consistency():
    """Property: Identical images have infinite PSNR, noisy images have finite dB."""
    img = torch.rand(1, 1, 64, 64)
    assert psnr(img, img) == float("inf")
    noisy = img + 0.1
    p = psnr(img, noisy, data_range=1.0)
    assert 15.0 < p < 25.0

def test_restoration_benchmark():
    """Validates the multi-algorithm restoration benchmark pipeline."""
    ref = torch.rand(1, 1, 64, 64)
    dist = add_gaussian_noise(ref, sigma=0.1)
    results = benchmark_restorations(ref, dist)
    assert len(results) >= 4
    assert results[0]["technique"] == "Baseline (Unrestored Distorted)"
