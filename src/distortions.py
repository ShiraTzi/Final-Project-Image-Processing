"""Image distortions + SNR — the three corruptions chosen in the main README.

  1. Gaussian noise     (sensor / random intensity noise)
  2. Salt-and-pepper     (sparse impulsive pixel corruption)
  3. Motion blur         (camera shake / object motion)

Implemented directly with numpy/cv2 so the benchmark is deterministic per
(image_id, distortion, severity) and independent of any augmentation library's
version. compute_snr() quantifies distortion intensity (used for acc-vs-SNR
curves).

Usage:
    python -m src.distortions --config configs/config.yaml      # build all distorted sets
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import load_config, ensure_dirs
from src.data import load_subset_ids

DISTORTIONS = ("gauss_noise", "salt_pepper", "motion_blur")


# --------------------------------------------------------------------------- #
# Core operations (deterministic given an explicit RNG)
# --------------------------------------------------------------------------- #
def gauss_noise(img_rgb: np.ndarray, var_limit, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise; variance sampled uniformly in ``var_limit`` (0-255 image)."""
    var = rng.uniform(float(var_limit[0]), float(var_limit[1]))
    sigma = float(np.sqrt(var))
    noise = rng.normal(0.0, sigma, img_rgb.shape)
    out = img_rgb.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def salt_pepper(img_rgb: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Set a fraction ``amount`` of pixels to pure white (salt) or black (pepper),
    split 50/50.  Affects all channels of the chosen pixels."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    n = int(amount * h * w)
    if n <= 0:
        return out
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    half = n // 2
    out[ys[:half], xs[:half]] = 255          # salt
    out[ys[half:], xs[half:]] = 0            # pepper
    return out


def motion_blur(img_rgb: np.ndarray, ksize: int, rng: np.random.Generator) -> np.ndarray:
    """Linear motion blur with a length-``ksize`` kernel at a random angle."""
    ksize = int(ksize)
    if ksize < 3:
        return img_rgb
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[(ksize - 1) // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    rot = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
    s = kernel.sum()
    kernel = kernel / s if s > 0 else kernel
    return cv2.filter2D(img_rgb, -1, kernel)


_FUNCS = {"gauss_noise": gauss_noise, "salt_pepper": salt_pepper, "motion_blur": motion_blur}


def apply_distortion(img_rgb: np.ndarray, dtype: str, params: Dict,
                     rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the right distortion using its severity params dict."""
    return _FUNCS[dtype](img_rgb, rng=rng, **params)


def _rng_for(seed: int, image_id: int, dtype: str, severity: str) -> np.random.Generator:
    """Deterministic per-image RNG so reruns reproduce identical distortions."""
    key = (seed, image_id, hash(dtype) & 0xFFFF, hash(severity) & 0xFFFF)
    return np.random.default_rng(abs(hash(key)) % (2**32))


def compute_snr(clean_rgb: np.ndarray, dist_rgb: np.ndarray) -> float:
    """SNR (dB) = 10*log10(signal_power / noise_power), noise = clean - distorted."""
    clean = clean_rgb.astype(np.float64)
    noise = clean - dist_rgb.astype(np.float64)
    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2))
    return 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")


# --------------------------------------------------------------------------- #
# Batch generation over the fixed val subset
# --------------------------------------------------------------------------- #
def build_distorted_sets(cfg: Dict) -> None:
    from pycocotools.coco import COCO

    ds = cfg["dataset"]
    seed = cfg["seed"]
    clean_dir = Path(cfg["paths"]["coco_root"]) / ds["val_split"]
    dist_root = Path(cfg["paths"]["distorted_root"])
    coco = COCO(ds["ann_instances_val"])
    img_ids = load_subset_ids(ds["val_subset_file"])
    id2name = {im["id"]: im["file_name"] for im in coco.loadImgs(img_ids)}

    snr_rows: List[Dict] = []
    for dtype, spec in cfg["distortions"].items():
        for severity, params in spec["severities"].items():
            out_dir = dist_root / dtype / severity
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_id in tqdm(img_ids, desc=f"{dtype}/{severity}", leave=False):
                fname = id2name[image_id]
                clean = np.array(Image.open(clean_dir / fname).convert("RGB"))
                rng = _rng_for(seed, image_id, dtype, severity)
                dist = apply_distortion(clean, dtype, params, rng)
                dst = out_dir / fname
                if not dst.exists():
                    Image.fromarray(dist).save(dst)
                snr_rows.append({
                    "image_id": image_id, "file_name": fname,
                    "distortion": dtype, "severity": severity,
                    "snr_db": round(compute_snr(clean, dist), 4),
                })

    snr_path = Path(cfg["paths"]["metrics_dir"]) / "snr_index.csv"
    snr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "file_name", "distortion", "severity", "snr_db"])
        w.writeheader()
        w.writerows(snr_rows)
    print(f"[distortions] wrote {len(snr_rows)} rows -> {snr_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    build_distorted_sets(cfg)


if __name__ == "__main__":
    main()
