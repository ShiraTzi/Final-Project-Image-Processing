"""Classical enhancement / restoration, one matched method per distortion.

  - gauss_noise  -> Non-Local Means + bilateral   (edge-preserving denoise)
  - salt_pepper  -> median filter                 (the classic impulse-noise remover)
  - motion_blur  -> unsharp masking               (sharpening as a simple deblur proxy)

All functions take and return an RGB uint8 array.

Usage:
    python -m src.enhancement --config configs/config.yaml   # enhance all distorted sets
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import load_config, ensure_dirs
from src.data import load_subset_ids


def restore_noise(img_rgb: np.ndarray) -> np.ndarray:
    """Gaussian-noise restoration: strong NLM denoise + edge-preserving bilateral."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    den = cv2.bilateralFilter(den, d=7, sigmaColor=50, sigmaSpace=50)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)


def restore_saltpepper(img_rgb: np.ndarray) -> np.ndarray:
    """Salt-and-pepper restoration: median filter (removes impulse pixels)."""
    return cv2.medianBlur(img_rgb, 3)


def restore_motionblur(img_rgb: np.ndarray) -> np.ndarray:
    """Motion-blur restoration: unsharp masking (sharpen high frequencies).

    A practical, blind deblurring proxy — we don't know the exact blur kernel at
    inference time, so we boost detail rather than deconvolve."""
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=3.0)
    sharp = cv2.addWeighted(img_rgb, 1.5, blurred, -0.5, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


RESTORERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gauss_noise": restore_noise,
    "salt_pepper": restore_saltpepper,
    "motion_blur": restore_motionblur,
}


def build_enhanced_sets(cfg: Dict) -> None:
    from pycocotools.coco import COCO

    ds = cfg["dataset"]
    dist_root = Path(cfg["paths"]["distorted_root"])
    enh_root = Path(cfg["paths"]["enhanced_root"])
    coco = COCO(ds["ann_instances_val"])
    img_ids = load_subset_ids(ds["val_subset_file"])
    id2name = {im["id"]: im["file_name"] for im in coco.loadImgs(img_ids)}

    for dtype, spec in cfg["distortions"].items():
        restorer = RESTORERS[dtype]
        for severity in spec["severities"]:
            in_dir = dist_root / dtype / severity
            out_dir = enh_root / dtype / severity
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_id in tqdm(img_ids, desc=f"{dtype}/{severity}", leave=False):
                fname = id2name[image_id]
                dst = out_dir / fname
                if dst.exists():
                    continue
                dist = np.array(Image.open(in_dir / fname).convert("RGB"))
                Image.fromarray(restorer(dist)).save(dst)
    print(f"[enhancement] enhanced sets written under {enh_root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    build_enhanced_sets(cfg)


if __name__ == "__main__":
    main()
