"""
SSIM LAB: Intelligent Structural Similarity & Image Quality Analysis Engine
"""

from .ssim import ssim, ms_ssim, SSIM, MS_SSIM
from .metrics import psnr, mse, mae, uqi, calculate_all_metrics, compute_gradient_perceptual_distance
from .distortions import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_gaussian_blur,
    apply_motion_blur,
    apply_real_jpeg_compression,
    apply_resizing_distortion,
    adjust_brightness_contrast,
)
from .restoration import (
    gaussian_filter,
    median_filter,
    bilateral_filter,
    non_local_means_denoising,
    unsharp_mask,
    benchmark_restorations,
)
from .visualizer import compute_diagnostic_maps, array_to_base64_image
from .experiments import save_experiment, list_experiments, get_experiment
from .reports import generate_markdown_report, generate_csv_report

__all__ = [
    "ssim",
    "ms_ssim",
    "SSIM",
    "MS_SSIM",
    "psnr",
    "mse",
    "mae",
    "uqi",
    "calculate_all_metrics",
    "compute_gradient_perceptual_distance",
    "add_gaussian_noise",
    "add_salt_and_pepper_noise",
    "apply_gaussian_blur",
    "apply_motion_blur",
    "apply_real_jpeg_compression",
    "apply_resizing_distortion",
    "adjust_brightness_contrast",
    "gaussian_filter",
    "median_filter",
    "bilateral_filter",
    "non_local_means_denoising",
    "unsharp_mask",
    "benchmark_restorations",
    "compute_diagnostic_maps",
    "array_to_base64_image",
    "save_experiment",
    "list_experiments",
    "get_experiment",
    "generate_markdown_report",
    "generate_csv_report",
]
