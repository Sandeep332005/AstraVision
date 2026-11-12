"""
SSIM LAB Formal Verification & Comprehensive Mathematical Test Suite
Validates mathematical axioms, perceptual metrics, real parametric distortions,
native restorations, diagnostic maps, experiment persistence, and report generation.
"""

import math
import io
import torch
import numpy as np
from PIL import Image
import pytest

from engine.ssim import ssim, ms_ssim, SSIM, MS_SSIM
from engine.metrics import calculate_all_metrics, psnr, mse, mae, uqi, perceptual_gradient_distance
from engine.distortions import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_gaussian_blur,
    apply_motion_blur,
    apply_real_jpeg_compression,
    apply_resizing_distortion,
    adjust_brightness_contrast,
)
from engine.restoration import (
    gaussian_filter,
    median_filter,
    bilateral_filter,
    non_local_means_denoising,
    unsharp_mask,
    benchmark_restorations,
)
from engine.visualizer import compute_diagnostic_maps, array_to_base64_image
from engine.experiments import save_experiment, list_experiments, get_experiment
from engine.reports import generate_markdown_report, generate_csv_report
from app.server import base64_to_tensor, tensor_to_base64


# ==============================================================================
# 1. MATHEMATICAL AXIOMS & CORE PROPERTIES
# ==============================================================================

def test_ssim_identity():
    """Property 1: SSIM(x, x) == 1.0 (Identical inputs yield perfect fidelity)."""
    torch.manual_seed(42)
    img = torch.rand(1, 1, 128, 128)
    score = ssim(img, img)
    assert abs(score.item() - 1.0) < 1e-4


def test_ssim_symmetry():
    """Property 2: SSIM(x, y) == SSIM(y, x) (Symmetric evaluation)."""
    torch.manual_seed(42)
    img1 = torch.rand(1, 1, 128, 128)
    img2 = torch.rand(1, 1, 128, 128)
    score1 = ssim(img1, img2)
    score2 = ssim(img2, img1)
    assert abs(score1.item() - score2.item()) < 1e-5


def test_data_range_invariance():
    """Property 3: SSIM on [0, 1] with L=1.0 must match SSIM on [0, 255] with L=255.0."""
    torch.manual_seed(42)
    img1_float = torch.rand(1, 1, 128, 128)
    img2_float = torch.rand(1, 1, 128, 128)

    score_float = ssim(img1_float, img2_float, data_range=1.0)

    img1_255 = img1_float * 255.0
    img2_255 = img2_float * 255.0
    score_255 = ssim(img1_255, img2_255, data_range=255.0)

    assert abs(score_float.item() - score_255.item()) < 1e-4


def test_ms_ssim_identity():
    """Property 4: MS-SSIM(x, x) == 1.0 across 5 dyadic pyramid scales."""
    torch.manual_seed(42)
    img = torch.rand(1, 1, 160, 160)
    score = ms_ssim(img, img)
    assert abs(score.item() - 1.0) < 1e-3


def test_multichannel_rgb_and_grayscale():
    """Validates SSIM and MS-SSIM on both 1-channel (grayscale) and 3-channel (RGB) images."""
    torch.manual_seed(42)
    gray = torch.rand(1, 1, 160, 160)
    rgb = torch.rand(1, 3, 160, 160)

    assert abs(ssim(gray, gray).item() - 1.0) < 1e-4
    assert abs(ssim(rgb, rgb).item() - 1.0) < 1e-4
    assert abs(ms_ssim(gray, gray).item() - 1.0) < 1e-3
    assert abs(ms_ssim(rgb, rgb).item() - 1.0) < 1e-3


def test_ssim_nn_module():
    """Validates PyTorch nn.Module wrappers SSIM() and MS_SSIM()."""
    torch.manual_seed(42)
    img = torch.rand(1, 3, 160, 160)
    ssim_module = SSIM(data_range=1.0)
    ms_ssim_module = MS_SSIM(data_range=1.0)

    assert abs(ssim_module(img, img).item() - 1.0) < 1e-4
    assert abs(ms_ssim_module(img, img).item() - 1.0) < 1e-3


def test_roi_regional_ssim():
    """Validates Region of Interest (ROI) bounding box evaluation."""
    torch.manual_seed(42)
    ref = torch.zeros(1, 1, 100, 100)
    dist = torch.zeros(1, 1, 100, 100)

    # Corrupt only the bottom-right quadrant (y: 50-100, x: 50-100)
    dist[:, :, 50:100, 50:100] = 1.0

    # Clean top-left region safely outside the window radius: SSIM is exactly 1.0
    score_clean_roi = ssim(ref, dist, roi=(0, 0, 40, 40)).item()
    assert abs(score_clean_roi - 1.0) < 1e-4

    # Corrupted bottom-right region: SSIM should be near 0
    score_dirty_roi = ssim(ref, dist, roi=(60, 60, 100, 100)).item()
    assert score_dirty_roi < 0.05


# ==============================================================================
# 2. METRICS ENGINE & PERCEPTUAL QUALITY
# ==============================================================================

def test_degradation_ordering():
    """Property: SSIM must drop strictly monotonically as noise intensity increases."""
    torch.manual_seed(42)
    ref = torch.rand(1, 1, 128, 128)
    mild_noise = add_gaussian_noise(ref, sigma=0.05)
    severe_noise = add_gaussian_noise(ref, sigma=0.30)

    score_mild = ssim(ref, mild_noise).item()
    score_severe = ssim(ref, severe_noise).item()

    assert score_mild > score_severe
    assert score_mild < 1.0


def test_psnr_consistency():
    """Property: Identical images have infinite PSNR, noisy images have finite positive dB."""
    img = torch.rand(1, 1, 64, 64)
    assert psnr(img, img) == float("inf")
    noisy = img + 0.1
    p = psnr(img, noisy, data_range=1.0)
    assert 15.0 < p < 25.0


def test_calculate_all_metrics():
    """Validates comprehensive metric computation dictionary."""
    torch.manual_seed(42)
    ref = torch.rand(1, 3, 160, 160)
    dist = add_gaussian_noise(ref, sigma=0.1)

    metrics = calculate_all_metrics(ref, dist)
    expected_keys = {"ssim", "ms_ssim", "psnr_db", "mse", "mae", "uqi", "perceptual_distance"}
    assert expected_keys.issubset(metrics.keys())
    assert 0.0 < metrics["ssim"] < 1.0
    assert 0.0 < metrics["ms_ssim"] < 1.0
    assert metrics["psnr_db"] > 0
    assert metrics["mse"] > 0
    assert metrics["mae"] > 0
    assert 0.0 < metrics["uqi"] <= 1.0
    assert metrics["perceptual_distance"] >= 0.0


def test_perceptual_gradient_distance():
    """Validates directional Sobel gradient perceptual distance."""
    torch.manual_seed(42)
    ref = torch.rand(1, 3, 64, 64)
    blurred = apply_gaussian_blur(ref, sigma=2.0)

    dist_identical = perceptual_gradient_distance(ref, ref)
    dist_blurred = perceptual_gradient_distance(ref, blurred)

    assert dist_identical == 0.0
    assert dist_blurred > 0.0


# ==============================================================================
# 3. PARAMETRIC DISTORTION LABORATORY
# ==============================================================================

def test_real_jpeg_compression():
    """Verifies genuine DCT JPEG quantization: quality=10 produces lower SSIM than quality=90."""
    x = torch.linspace(0, 1, 128)
    y = torch.linspace(0, 1, 128)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    ref = (torch.sin(grid_x * 6.28) * 0.4 + 0.5).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    q90 = apply_real_jpeg_compression(ref, quality=90)
    q10 = apply_real_jpeg_compression(ref, quality=10)

    ssim_q90 = ssim(ref, q90).item()
    ssim_q10 = ssim(ref, q10).item()

    assert ssim_q90 > ssim_q10
    assert ssim_q90 > 0.95
    assert ssim_q10 < ssim_q90


def test_salt_and_pepper_reproducibility():
    """Validates salt and pepper noise generation and reproducibility with seeds."""
    ref = torch.full((1, 1, 64, 64), 0.5)
    sp1 = add_salt_and_pepper_noise(ref, prob=0.1, seed=123)
    sp2 = add_salt_and_pepper_noise(ref, prob=0.1, seed=123)
    sp3 = add_salt_and_pepper_noise(ref, prob=0.1, seed=999)

    assert torch.equal(sp1, sp2)
    assert not torch.equal(sp1, sp3)
    assert torch.any(sp1 == 0.0)
    assert torch.any(sp1 == 1.0)


def test_motion_blur():
    """Validates directional motion blur attenuation."""
    ref = torch.rand(1, 3, 64, 64)
    blurred = apply_motion_blur(ref, size=9, angle=45.0)

    assert blurred.shape == ref.shape
    assert ssim(ref, blurred).item() < 1.0


def test_brightness_contrast():
    """Validates brightness and contrast adjustments."""
    ref = torch.rand(1, 3, 64, 64)
    bright = adjust_brightness_contrast(ref, brightness=0.2, contrast=1.2)

    assert bright.shape == ref.shape
    assert bright.min() >= 0.0
    assert bright.max() <= 1.0


def test_resize_distortion():
    """Validates decimation and bilinear upsampling distortion."""
    ref = torch.rand(1, 3, 64, 64)
    resized = apply_resizing_distortion(ref, scale_factor=0.25)

    assert resized.shape == ref.shape
    assert ssim(ref, resized).item() < 0.95


# ==============================================================================
# 4. IMAGE RESTORATION LAB & BENCHMARKING
# ==============================================================================

def test_restoration_benchmark():
    """Validates the multi-algorithm restoration benchmark pipeline."""
    ref = torch.rand(1, 3, 64, 64)
    dist = add_gaussian_noise(ref, sigma=0.1)
    results = benchmark_restorations(ref, dist)

    assert len(results) >= 5
    assert results[0]["algorithm"] == "Baseline (Unrestored Distorted)"
    for r in results:
        assert "algorithm" in r
        assert "ssim" in r
        assert "ms_ssim" in r
        assert "psnr_db" in r
        assert "mse" in r
        assert "delta_ssim" in r
        assert "runtime_ms" in r
        assert "memory_kb" in r


def test_non_local_means_denoising():
    """Validates native PyTorch Non-Local Means (NLM) patch denoising."""
    ref = torch.rand(1, 3, 48, 48)
    noisy = add_gaussian_noise(ref, sigma=0.1)
    denoised = non_local_means_denoising(noisy, h=0.08)

    assert denoised.shape == noisy.shape
    assert denoised.min() >= 0.0
    assert denoised.max() <= 1.0


def test_bilateral_filter():
    """Validates edge-preserving bilateral filtering."""
    ref = torch.rand(1, 3, 48, 48)
    filtered = bilateral_filter(ref, kernel_size=5, sigma_s=2.5, sigma_r=0.15)

    assert filtered.shape == ref.shape
    assert filtered.min() >= 0.0
    assert filtered.max() <= 1.0


# ==============================================================================
# 5. DIAGNOSTIC HEATMAPS & VISUALIZATION
# ==============================================================================

def test_diagnostic_maps():
    """Validates calculation of local SSIM, error, edge discrepancy, and threshold masks."""
    ref = torch.rand(1, 3, 64, 64)
    dist = add_gaussian_noise(ref, sigma=0.1)

    maps = compute_diagnostic_maps(ref, dist, threshold=0.85)

    assert "ssim_map" in maps
    assert "error_map" in maps
    assert "edge_diff_map" in maps
    assert "threshold_mask" in maps

    assert maps["ssim_map"].shape == (64, 64)
    assert maps["error_map"].shape == (64, 64)
    assert maps["edge_diff_map"].shape == (64, 64)
    assert maps["threshold_mask"].shape == (64, 64)

    # Base64 export test
    b64 = array_to_base64_image(maps["ssim_map"], colormap="jet")
    assert b64.startswith("data:image/png;base64,")


# ==============================================================================
# 6. EXPERIMENT MANAGEMENT & SCIENTIFIC REPORTS
# ==============================================================================

def test_experiment_persistence():
    """Validates saving, listing, and retrieving experimental runs."""
    base_metrics = {"ssim": 0.72, "psnr_db": 22.4}
    rest_metrics = {"ssim": 0.81, "psnr_db": 25.1}

    record = save_experiment(
        name="PyTest Unit Experiment",
        distortion_type="Gaussian Noise",
        distortion_params={"sigma": 0.15},
        restoration_method="Non-Local Means",
        baseline_metrics=base_metrics,
        restored_metrics=rest_metrics,
        seed=42,
        image_name="test_card.png",
    )

    assert "id" in record
    assert record["delta_ssim"] == round(0.81 - 0.72, 4)

    all_exps = list_experiments()
    assert any(e["id"] == record["id"] for e in all_exps)

    fetched = get_experiment(record["id"])
    assert fetched is not None
    assert fetched["name"] == "PyTest Unit Experiment"


def test_report_generation():
    """Validates Markdown and CSV scientific report generation."""
    exp = {
        "id": "EXP-TEST-001",
        "timestamp": "2026-11-12 12:00:00",
        "image_name": "sample.png",
        "seed": 42,
        "distortion": {"type": "JPEG", "params": {"quality": 20}},
    }
    benchmarks = [
        {
            "algorithm": "Baseline",
            "ssim": 0.75,
            "ms_ssim": 0.78,
            "psnr_db": 24.5,
            "mse": 120.0,
            "delta_ssim": 0.0,
            "runtime_ms": 1.2,
            "memory_kb": 192.0,
            "status": "Baseline",
        },
        {
            "algorithm": "Median Filter",
            "ssim": 0.82,
            "ms_ssim": 0.84,
            "psnr_db": 26.1,
            "mse": 95.0,
            "delta_ssim": 0.07,
            "runtime_ms": 4.5,
            "memory_kb": 192.0,
            "status": "Improved",
        },
    ]
    meta = {"width": 256, "height": 256, "channels": 3}

    # Markdown verification
    md = generate_markdown_report(exp, benchmarks, meta)
    assert "# Research Report: Perceptual Image Quality & Structural Similarity Analysis" in md
    assert "Median Filter" in md
    assert "+0.0700" in md

    # CSV verification
    csv_str = generate_csv_report(benchmarks)
    assert "Algorithm,SSIM,MS-SSIM,PSNR_dB,MSE,Delta_SSIM,Runtime_ms,Memory_KB,Status" in csv_str
    assert "Median Filter,0.82,0.84,26.1,95.0,0.07,4.5,192.0,Improved" in csv_str


# ==============================================================================
# 7. SERVER DATA SERIALIZATION FIDELITY
# ==============================================================================

def test_base64_tensor_roundtrip():
    """Validates base64 data URL encode and decode roundtrip."""
    img_tensor = torch.zeros(1, 3, 32, 32)
    img_tensor[:, 0, :, :] = 0.8  # Red
    img_tensor[:, 1, :, :] = 0.4  # Green

    b64 = tensor_to_base64(img_tensor)
    assert b64.startswith("data:image/png;base64,")

    decoded_tensor, dims = base64_to_tensor(b64)
    assert dims == (32, 32)
    assert decoded_tensor.shape == (1, 3, 32, 32)
    # RGB roundtrip error should be within 1/255 quantization error
    diff = torch.abs(img_tensor - decoded_tensor)
    assert diff.max().item() < (2.0 / 255.0)
