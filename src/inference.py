"""Run a (task, variant) and write COCO-format prediction JSON.

Tasks handled here: ``detection`` (YOLOv8, bbox) and ``keypoints``
(torchvision Keypoint R-CNN).  Panoptic segmentation runs separately in
src/segmentation.py (detectron2 venv).

Predictions are written to results/preds/{task}__{variant_tag}.json so the
single COCOeval path in metrics.py can consume them.

Usage:
    python -m src.inference --task detection --variant clean
    python -m src.inference --task keypoints --variant distorted --dtype gauss_noise --severity high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.config import load_config, ensure_dirs, variant_image_dir, variant_tag
from src.data import load_subset_ids
from src.models import (
    COCO80_TO_91, get_keypoint_model, get_yolo_model, resolve_device, yolo_device,
)


def _subset_images(cfg: Dict, variant: str, dtype, severity):
    """Yield (image_id, file_path) for the fixed val subset in the chosen variant dir."""
    from pycocotools.coco import COCO

    ds = cfg["dataset"]
    coco = COCO(ds["ann_instances_val"])
    img_ids = load_subset_ids(ds["val_subset_file"])
    img_dir = variant_image_dir(cfg, variant, dtype, severity)
    for im in coco.loadImgs(img_ids):
        yield im["id"], img_dir / im["file_name"]


def _run_yolo(model, items, score_thr, imgsz, device) -> List[Dict]:
    """YOLOv8 detection -> COCO-format dicts (class idx 0-79 -> COCO category id)."""
    results: List[Dict] = []
    for image_id, path in tqdm(items, desc="detection", leave=False):
        r = model.predict(str(path), conf=score_thr, imgsz=imgsz, device=device,
                          verbose=False)[0]
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, c in zip(xyxy, conf, cls):
            results.append({
                "image_id": int(image_id),
                "category_id": int(COCO80_TO_91[c]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    return results


@torch.no_grad()
def _run_keypoints(model, transform, items, device, score_thr) -> List[Dict]:
    results: List[Dict] = []
    for image_id, path in tqdm(items, desc="keypoints", leave=False):
        img = Image.open(path).convert("RGB")
        out = model([transform(img).to(device)])[0]
        scores = out["scores"].cpu().numpy()
        kpts = out["keypoints"].cpu().numpy()              # [N,17,3]
        keep = scores >= score_thr
        for kp, score in zip(kpts[keep], scores[keep]):
            flat: List[float] = []
            for x, y, _ in kp:
                flat += [float(x), float(y), 1]            # v=1; OKS uses gt visibility
            results.append({
                "image_id": int(image_id),
                "category_id": 1,                           # person
                "keypoints": flat,
                "score": float(score),
            })
    return results


def run_inference(cfg: Dict, task: str, variant: str, dtype=None, severity=None,
                  checkpoint: str = None) -> str:
    score_thr = cfg["inference"]["score_threshold"]
    items = list(_subset_images(cfg, variant, dtype, severity))

    if task == "detection":
        weights = checkpoint or cfg["yolo"]["weights"]
        model = get_yolo_model(weights)
        if checkpoint:
            print(f"[inference] using fine-tuned YOLO checkpoint {checkpoint}")
        results = _run_yolo(model, items, score_thr, cfg["yolo"]["imgsz"],
                            yolo_device(cfg["inference"]["device"]))
    elif task == "keypoints":
        device = resolve_device(cfg["inference"]["device"])
        model, transform, _ = get_keypoint_model(device)
        results = _run_keypoints(model, transform, items, device, score_thr)
    else:
        raise ValueError(f"inference.py does not handle task={task} (ORB is in metrics.py)")

    tag = variant_tag(variant, dtype, severity)
    out_path = Path(cfg["paths"]["preds_dir"]) / f"{task}__{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[inference] {task}/{tag}: {len(results)} detections -> {out_path}")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", required=True, choices=["detection", "keypoints"])
    ap.add_argument("--variant", required=True, choices=["clean", "distorted", "enhanced"])
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--severity", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.variant != "clean" and (not args.dtype or not args.severity):
        ap.error("--dtype and --severity are required for distorted/enhanced variants")
    run_inference(cfg, args.task, args.variant, args.dtype, args.severity)


if __name__ == "__main__":
    main()
