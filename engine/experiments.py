"""
Experiment Management System
Tracks, persists, and compares scientific image quality experiments.
Supports reproducibility with saved seeds, configurations, and metric logs.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

EXPERIMENTS_FILE = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "experiments_log.json")

def _load_all() -> List[Dict[str, Any]]:
    os.makedirs(os.path.dirname(EXPERIMENTS_FILE), exist_ok=True)
    if not os.path.exists(EXPERIMENTS_FILE):
        return []
    try:
        with open(EXPERIMENTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def _save_all(data: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(EXPERIMENTS_FILE), exist_ok=True)
    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def save_experiment(
    name: str,
    distortion_type: str,
    distortion_params: Dict[str, Any],
    restoration_method: str,
    baseline_metrics: Dict[str, Any],
    restored_metrics: Dict[str, Any],
    seed: Optional[int] = None,
    image_name: str = "Uploaded Image",
) -> Dict[str, Any]:
    """Records an experiment with full reproducibility metadata."""
    experiments = _load_all()
    exp_id = f"EXP-{int(time.time())}-{len(experiments) + 1}"
    record = {
        "id": exp_id,
        "name": name or f"Experiment {len(experiments) + 1}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "seed": seed,
        "distortion": {
            "type": distortion_type,
            "params": distortion_params,
        },
        "restoration": {
            "method": restoration_method,
        },
        "baseline_metrics": baseline_metrics,
        "restored_metrics": restored_metrics,
        "delta_ssim": round(restored_metrics.get("ssim", 0.0) - baseline_metrics.get("ssim", 0.0), 4),
        "delta_psnr": round(restored_metrics.get("psnr_db", 0.0) - baseline_metrics.get("psnr_db", 0.0), 2),
    }
    experiments.insert(0, record)
    _save_all(experiments)
    return record

def list_experiments() -> List[Dict[str, Any]]:
    """Returns all recorded experiment summaries."""
    return _load_all()

def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific experiment by ID."""
    for exp in _load_all():
        if exp["id"] == exp_id:
            return exp
    return None

def delete_experiment(exp_id: str) -> bool:
    """Deletes an experiment by ID."""
    exps = _load_all()
    filtered = [e for e in exps if e["id"] != exp_id]
    if len(filtered) != len(exps):
        _save_all(filtered)
        return True
    return False
