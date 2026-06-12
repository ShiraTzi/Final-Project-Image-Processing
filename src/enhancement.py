"""Classical enhancement / restoration, one matched method per distortion.

Lifted from the reference pipeline (§3.1):
  - gauss_noise -> Non-Local Means + bilateral
  - severe_jpeg -> Y-channel bilateral filtering
  - low_light   -> gamma correction + CLAHE

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
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 25, 25, 7, 35)
    den = cv2.bilateralFilter(den, d=9, sigmaColor=80, sigmaSpace=80)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)


def restore_jpeg(img_rgb: np.ndarray) -> np.ndarray:
    """Severe-JPEG restoration: bilateral on the Y (luma) channel only."""
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.bilateralFilter(y, d=7, sigmaColor=40, sigmaSpace=40)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)


def restore_lowlight(img_rgb: np.ndarray) -> np.ndarray:
    """Low-light restoration: gamma lift + CLAHE local contrast on L channel."""
    gamma = 0.35
    lut = (np.arange(256) / 255.0) ** gamma * 255
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    img_gamma = cv2.LUT(img_rgb, lut)

    lab = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


RESTORERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gauss_noise": restore_noise,
    "severe_jpeg": restore_jpeg,
    "low_light": restore_lowlight,
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
