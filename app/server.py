"""
SSIM LAB High-Performance Server & REST API
Zero-dependency HTTP server bridging the scientific PyTorch engine with the scientific workspace.
Endpoints for analysis, real distortions, restoration, benchmarking, experiments, and research reports.
"""

import os
import sys
import json
import base64
import io
import time
from typing import Tuple, Dict, Any, Optional
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
import torch
import numpy as np
from PIL import Image

# Add root directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.ssim import ssim, ms_ssim
from engine.metrics import calculate_all_metrics
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
from engine.experiments import save_experiment, list_experiments, get_experiment, delete_experiment
from engine.reports import generate_markdown_report, generate_csv_report

def base64_to_tensor(b64_str: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Decodes a base64 image data URL into a PyTorch tensor (1, C, H, W) normalized to [0, 1]."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = pil_img.size
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    return tensor, (W, H)

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Converts a PyTorch tensor [0, 1] into a base64 PNG data URL."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    np_img = (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

class SSIMLabRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        static_dir = os.path.join(BASE_DIR, "app", "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)
        try:
            data = json.loads(post_body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON format"}, status=400)
            return

        routes = {
            "/api/analyze": self._handle_analyze,
            "/api/distort": self._handle_distort,
            "/api/restore": self._handle_restore,
            "/api/benchmark": self._handle_benchmark,
            "/api/experiments/save": self._handle_save_experiment,
            "/api/experiments/list": self._handle_list_experiments,
            "/api/experiments/delete": self._handle_delete_experiment,
            "/api/reports/export": self._handle_export_report,
        }

        handler = routes.get(self.path)
        if handler:
            handler(data)
        else:
            self._send_json({"error": f"Endpoint {self.path} not found"}, status=404)

    def _send_json(self, response_data: dict, status: int = 200):
        body = json.dumps(response_data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            self._send_json({"status": "healthy", "service": "SSIM LAB Engine", "version": "2.0.0"})
            return
        elif self.path == "/api/experiments":
            records = list_experiments()
            self._send_json({"experiments": records})
            return
        super().do_GET()

    def _handle_analyze(self, data: dict):
        try:
            ref, (w_ref, h_ref) = base64_to_tensor(data["ref"])
            dist, (w_dist, h_dist) = base64_to_tensor(data["dist"])

            # Resize distorted image to match reference if dimensions mismatch
            if ref.shape[-2:] != dist.shape[-2:]:
                dist = torch.nn.functional.interpolate(dist, size=ref.shape[-2:], mode="bilinear", align_corners=False)

            roi = data.get("roi") # [ymin, xmin, ymax, xmax]
            threshold = float(data.get("threshold", 0.80))
            overlay_alpha = float(data.get("overlay_alpha", 0.50))

            metrics = calculate_all_metrics(ref, dist, roi=roi)
            maps = compute_diagnostic_maps(ref, dist, threshold=threshold, overlay_alpha=overlay_alpha)

            # Convert ref for overlay
            ref_np = ref.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

            response = {
                "metrics": metrics,
                "heatmaps": {
                    "ssim_map": array_to_base64_image(maps["ssim_map"], colormap="jet"),
                    "error_map": array_to_base64_image(maps["error_map"], colormap="gray"),
                    "edge_diff_map": array_to_base64_image(maps["edge_diff_map"], colormap="gray"),
                    "ref_edges": array_to_base64_image(maps["ref_edges"], colormap="gray"),
                    "dist_edges": array_to_base64_image(maps["dist_edges"], colormap="gray"),
                    "threshold_mask": array_to_base64_image(maps["threshold_mask"], colormap="mask", ref_bg=ref_np, alpha=0.7),
                },
                "histogram": maps.get("histogram", {}),
                "image_meta": {
                    "width": w_ref,
                    "height": h_ref,
                    "channels": 3,
                }
            }
            self._send_json(response)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_distort(self, data: dict):
        try:
            img, _ = base64_to_tensor(data["image"])
            d_type = data.get("type", "blur")
            params = data.get("params", {})
            seed = params.get("seed")
            if seed is not None:
                try: seed = int(seed)
                except ValueError: seed = None

            if d_type == "blur":
                dist = apply_gaussian_blur(img, sigma=float(params.get("sigma", 2.5)))
            elif d_type == "noise":
                dist = add_gaussian_noise(img, sigma=float(params.get("sigma", 0.15)), seed=seed)
            elif d_type == "salt_pepper":
                dist = add_salt_and_pepper_noise(img, prob=float(params.get("prob", 0.05)), seed=seed)
            elif d_type == "motion":
                dist = apply_motion_blur(img, size=int(params.get("size", 15)), angle=float(params.get("angle", 45.0)))
            elif d_type == "jpeg":
                dist = apply_real_jpeg_compression(img, quality=int(params.get("quality", 15)))
            elif d_type == "resize":
                dist = apply_resizing_distortion(img, scale_factor=float(params.get("scale", 0.25)))
            elif d_type == "brightness_contrast":
                dist = adjust_brightness_contrast(img, brightness=float(params.get("brightness", 0.0)), contrast=float(params.get("contrast", 1.0)))
            else:
                dist = img.clone()

            metrics = calculate_all_metrics(img, dist)
            self._send_json({
                "distorted_image": tensor_to_base64(dist),
                "metrics": metrics,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_restore(self, data: dict):
        try:
            ref, _ = base64_to_tensor(data["ref"])
            dist, _ = base64_to_tensor(data["dist"])
            if ref.shape[-2:] != dist.shape[-2:]:
                dist = torch.nn.functional.interpolate(dist, size=ref.shape[-2:], mode="bilinear", align_corners=False)

            method = data.get("method", "median")
            params = data.get("params", {})

            t0 = time.perf_counter_ns()
            if method == "median":
                restored = median_filter(dist, kernel_size=int(params.get("kernel_size", 3)))
            elif method == "gaussian":
                restored = gaussian_filter(dist, sigma=float(params.get("sigma", 1.2)))
            elif method == "bilateral":
                restored = bilateral_filter(dist, kernel_size=5, sigma_s=2.5, sigma_r=0.15)
            elif method == "nlm":
                restored = non_local_means_denoising(dist, h=float(params.get("h", 0.08)))
            elif method == "unsharp":
                restored = unsharp_mask(dist, sigma=1.0, amount=float(params.get("amount", 0.8)))
            else:
                restored = dist.clone()

            runtime_ms = (time.perf_counter_ns() - t0) / 1e6
            base_ssim = ssim(ref, dist).item()
            new_metrics = calculate_all_metrics(ref, restored)
            delta = new_metrics["ssim"] - base_ssim

            self._send_json({
                "restored_image": tensor_to_base64(restored),
                "metrics": new_metrics,
                "delta_ssim": round(delta, 4),
                "runtime_ms": round(runtime_ms, 2),
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_benchmark(self, data: dict):
        try:
            ref, _ = base64_to_tensor(data["ref"])
            dist, _ = base64_to_tensor(data["dist"])
            if ref.shape[-2:] != dist.shape[-2:]:
                dist = torch.nn.functional.interpolate(dist, size=ref.shape[-2:], mode="bilinear", align_corners=False)

            # Keep benchmark responsive on large mega-pixel images
            if max(ref.shape[-2:]) > 768:
                scale = 768.0 / max(ref.shape[-2:])
                target_h = int(ref.shape[-2] * scale)
                target_w = int(ref.shape[-1] * scale)
                ref_bm = torch.nn.functional.interpolate(ref, size=(target_h, target_w), mode="bilinear", align_corners=False)
                dist_bm = torch.nn.functional.interpolate(dist, size=(target_h, target_w), mode="bilinear", align_corners=False)
            else:
                ref_bm = ref
                dist_bm = dist

            results = benchmark_restorations(ref_bm, dist_bm)
            self._send_json({"benchmark_results": results})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_save_experiment(self, data: dict):
        try:
            record = save_experiment(
                name=data.get("name", "Untitled Experiment"),
                distortion_type=data.get("distortion_type", "None"),
                distortion_params=data.get("distortion_params", {}),
                restoration_method=data.get("restoration_method", "None"),
                baseline_metrics=data.get("baseline_metrics", {}),
                restored_metrics=data.get("restored_metrics", {}),
                seed=data.get("seed"),
                image_name=data.get("image_name", "Uploaded Image"),
            )
            self._send_json({"status": "success", "record": record})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_list_experiments(self, data: dict):
        try:
            records = list_experiments()
            self._send_json({"experiments": records})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_delete_experiment(self, data: dict):
        try:
            exp_id = data.get("id")
            success = delete_experiment(exp_id)
            self._send_json({"status": "success" if success else "not_found"})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_export_report(self, data: dict):
        try:
            fmt = data.get("format", "markdown")
            exp_data = data.get("experiment", {})
            benchmarks = data.get("benchmarks", [])
            meta = data.get("image_meta", {})

            if fmt == "csv":
                content = generate_csv_report(benchmarks)
                self._send_json({"content": content, "mime": "text/csv", "filename": "ssim_benchmark_report.csv"})
            else:
                content = generate_markdown_report(exp_data, benchmarks, meta)
                self._send_json({"content": content, "mime": "text/markdown", "filename": "ssim_research_report.md"})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

def start_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, SSIMLabRequestHandler)
    print(f"==================================================")
    print(f"🔬 SSIM LAB Server running at: http://localhost:{port}")
    print(f"   Architecture: Track A Unified Scientific Studio")
    print(f"   Engine: PyTorch 2.x (CUDA/MPS/CPU Auto-Dispatch)")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SSIM LAB server...")
        httpd.server_close()

if __name__ == "__main__":
    start_server(8000)
