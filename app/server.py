"""
SSIM LAB High-Performance Server & REST API
Zero-dependency HTTP server bridging the scientific PyTorch engine with the frontend workspace.
"""

import os
import sys
import json
import base64
import io
from http.server import HTTPServer, SimpleHTTPRequestHandler
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
    apply_jpeg_artifacts,
    adjust_brightness_contrast,
)
from engine.restoration import (
    gaussian_filter,
    median_filter,
    bilateral_filter,
    unsharp_mask,
    benchmark_restorations,
)
from engine.visualizer import compute_diagnostic_maps, array_to_base64_image

def base64_to_tensor(b64_str: str) -> torch.Tensor:
    """Decodes a base64 image data URL into a PyTorch tensor (1, C, H, W) normalized to [0, 1]."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    return tensor

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Converts a (1, C, H, W) or (C, H, W) PyTorch tensor [0, 1] into a base64 PNG data URL."""
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

        if self.path == "/api/analyze":
            self._handle_analyze(data)
        elif self.path == "/api/distort":
            self._handle_distort(data)
        elif self.path == "/api/restore":
            self._handle_restore(data)
        elif self.path == "/api/benchmark":
            self._handle_benchmark(data)
        else:
            self._send_json({"error": f"Endpoint {self.path} not found"}, status=404)

    def _send_json(self, response_data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_analyze(self, data: dict):
        try:
            ref = base64_to_tensor(data["ref"])
            dist = base64_to_tensor(data["dist"])

            # Resize dist to ref shape if needed
            if ref.shape[-2:] != dist.shape[-2:]:
                dist = torch.nn.functional.interpolate(dist, size=ref.shape[-2:], mode="bilinear", align_corners=False)

            metrics = calculate_all_metrics(ref, dist)
            maps = compute_diagnostic_maps(ref, dist)

            response = {
                "metrics": metrics,
                "heatmaps": {
                    "ssim_map": array_to_base64_image(maps["ssim_map"], colormap="jet"),
                    "error_map": array_to_base64_image(maps["error_map"], colormap="hot"),
                    "edge_diff_map": array_to_base64_image(maps["edge_diff_map"], colormap="jet"),
                }
            }
            self._send_json(response)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_distort(self, data: dict):
        try:
            img = base64_to_tensor(data["image"])
            d_type = data.get("type", "blur")
            params = data.get("params", {})

            if d_type == "blur":
                dist = apply_gaussian_blur(img, sigma=float(params.get("sigma", 2.5)))
            elif d_type == "noise":
                dist = add_gaussian_noise(img, sigma=float(params.get("sigma", 0.15)))
            elif d_type == "salt_pepper":
                dist = add_salt_and_pepper_noise(img, prob=float(params.get("prob", 0.05)))
            elif d_type == "motion":
                dist = apply_motion_blur(img, size=int(params.get("size", 15)), angle=float(params.get("angle", 45)))
            elif d_type == "jpeg":
                dist = apply_jpeg_artifacts(img, block_size=int(params.get("block_size", 8)))
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
            ref = base64_to_tensor(data["ref"])
            dist = base64_to_tensor(data["dist"])
            method = data.get("method", "median")
            params = data.get("params", {})

            if method == "median":
                restored = median_filter(dist, kernel_size=int(params.get("kernel_size", 3)))
            elif method == "gaussian":
                restored = gaussian_filter(dist, sigma=float(params.get("sigma", 1.2)))
            elif method == "bilateral":
                restored = bilateral_filter(dist, kernel_size=5, sigma_s=2.5, sigma_r=0.15)
            elif method == "unsharp":
                restored = unsharp_mask(dist, sigma=1.0, amount=float(params.get("amount", 0.8)))
            else:
                restored = dist.clone()

            base_ssim = ssim(ref, dist).item()
            new_metrics = calculate_all_metrics(ref, restored)
            delta = new_metrics["ssim"] - base_ssim

            self._send_json({
                "restored_image": tensor_to_base64(restored),
                "metrics": new_metrics,
                "delta_ssim": round(delta, 4),
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_benchmark(self, data: dict):
        try:
            ref = base64_to_tensor(data["ref"])
            dist = base64_to_tensor(data["dist"])
            results = benchmark_restorations(ref, dist)
            self._send_json({"benchmark_results": results})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

def start_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, SSIMLabRequestHandler)
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
