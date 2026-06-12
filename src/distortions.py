"""Image distortions + SNR, mirroring the reference pipeline's three corruptions.

The reference used albumentations (`GaussNoise`, `ImageCompression`,
`RandomBrightnessContrast`).  Albumentations' GaussNoise signature changed
across releases (``var_limit`` was removed in 2.0), which would make a
reproducible benchmark fragile, so we implement the same three operations
directly with numpy/cv2.  Behaviour is equivalent and fully deterministic
per (image_id, distortion, severity).

compute_snr() is taken verbatim from the reference (§2.5).

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

DISTORTIONS = ("gauss_noise", "severe_jpeg", "low_light")


# --------------------------------------------------------------------------- #
# Core operations (deterministic given an explicit RNG)
# --------------------------------------------------------------------------- #
def gauss_noise(img_rgb: np.ndarray, var_limit, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise; variance sampled uniformly in ``var_limit``
    (matches albumentations.GaussNoise on a 0-255 image)."""
    var = rng.uniform(float(var_limit[0]), float(var_limit[1]))
    sigma = float(np.sqrt(var))
    noise = rng.normal(0.0, sigma, img_rgb.shape)
    out = img_rgb.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def severe_jpeg(img_rgb: np.ndarray, quality_lower: int, quality_upper: int,
                rng: np.random.Generator) -> np.ndarray:
    """Re-encode as JPEG at a low quality (matches albumentations.ImageCompression)."""
    q = int(rng.integers(quality_lower, quality_upper + 1))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img_rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def low_light(img_rgb: np.ndarray, brightness_limit, rng: np.random.Generator) -> np.ndarray:
    """Darken via additive brightness on max value (albumentations
    RandomBrightnessContrast default brightness_by_max=True, contrast=0):
    out = img + beta * 255."""
    beta = rng.uniform(float(brightness_limit[0]), float(brightness_limit[1]))
    out = img_rgb.astype(np.float32) + beta * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


_FUNCS = {"gauss_noise": gauss_noise, "severe_jpeg": severe_jpeg, "low_light": low_light}


def apply_distortion(img_rgb: np.ndarray, dtype: str, params: Dict,
                     rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the right distortion using its severity params dict."""
    return _FUNCS[dtype](img_rgb, rng=rng, **params)


def _rng_for(seed: int, image_id: int, dtype: str, severity: str) -> np.random.Generator:
    """Deterministic per-image RNG so reruns reproduce identical distortions."""
    key = (seed, image_id, hash(dtype) & 0xFFFF, hash(severity) & 0xFFFF)
    return np.random.default_rng(abs(hash(key)) % (2**32))


def compute_snr(clean_rgb: np.ndarray, dist_rgb: np.ndarray) -> float:
    """SNR (dB) = 10*log10(signal_power / noise_power), noise = clean - distorted.
    Verbatim from reference §2.5."""
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
                src = clean_dir / fname
                dst = out_dir / fname
                clean = np.array(Image.open(src).convert("RGB"))
                rng = _rng_for(seed, image_id, dtype, severity)
                dist = apply_distortion(clean, dtype, params, rng)
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
