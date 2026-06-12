"""Evaluation: COCOeval for detection/keypoints + ORB match ratio.

One metric file per (task, variant) under results/metrics/{task}__{tag}.json,
so tables.py / visualize.py can assemble degradation tables and SNR curves.

Usage:
    python -m src.metrics --task detection --variant clean
    python -m src.metrics --task orb       --variant distorted --dtype low_light --severity high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import load_config, ensure_dirs, variant_image_dir, variant_tag
from src.data import load_subset_ids
from src.models import orb_match_ratio


# --------------------------------------------------------------------------- #
# COCOeval (detection / keypoints)
# --------------------------------------------------------------------------- #
def _coco_eval(ann_path: str, pred_path: str, img_ids: List[int], iou_type: str):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_path)
    with open(pred_path) as f:
        preds = json.load(f)
    if not preds:
        return None, coco_gt
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev, coco_gt


def _per_class_ap(ev, coco_gt) -> Dict[str, float]:
    """Per-class AP@[.5:.95] from the precision tensor [T,R,K,A,M]."""
    precision = ev.eval["precision"]  # iou x recall x cat x area x maxdet
    cat_ids = ev.params.catIds
    out: Dict[str, float] = {}
    for k, cid in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]          # all-area, max dets
        p = p[p > -1]
        ap = float(np.mean(p)) if p.size else float("nan")
        name = coco_gt.loadCats([cid])[0]["name"]
        out[name] = round(ap, 4)
    return out


def evaluate_coco(cfg: Dict, task: str, variant: str, dtype=None, severity=None) -> Dict:
    ds = cfg["dataset"]
    img_ids = load_subset_ids(ds["val_subset_file"])
    tag = variant_tag(variant, dtype, severity)
    pred_path = Path(cfg["paths"]["preds_dir"]) / f"{task}__{tag}.json"

    if task == "detection":
        ann, iou_type = ds["ann_instances_val"], "bbox"
    elif task == "keypoints":
        ann, iou_type = ds["ann_keypoints_val"], "keypoints"
    else:
        raise ValueError(task)

    ev, coco_gt = _coco_eval(ann, str(pred_path), img_ids, iou_type)
    if ev is None:
        return {"task": task, "variant": tag, "empty": True, "mAP": 0.0}

    stats = ev.stats.tolist()
    result = {
        "task": task, "variant": tag, "iou_type": iou_type,
        "mAP": round(stats[0], 4), "mAP_50": round(stats[1], 4), "mAP_75": round(stats[2], 4),
        "stats": [round(s, 4) for s in stats],
    }
    if task == "detection":
        result["mAP_small"] = round(stats[3], 4)
        result["mAP_medium"] = round(stats[4], 4)
        result["mAP_large"] = round(stats[5], 4)
        result["per_class_ap"] = _per_class_ap(ev, coco_gt)
    return result


# --------------------------------------------------------------------------- #
# ORB match ratio (clean vs variant)
# --------------------------------------------------------------------------- #
def evaluate_orb(cfg: Dict, variant: str, dtype=None, severity=None) -> Dict:
    from pycocotools.coco import COCO

    ds = cfg["dataset"]
    coco = COCO(ds["ann_instances_val"])
    img_ids = load_subset_ids(ds["val_subset_file"])
    id2name = {im["id"]: im["file_name"] for im in coco.loadImgs(img_ids)}

    clean_dir = variant_image_dir(cfg, "clean")
    var_dir = variant_image_dir(cfg, variant, dtype, severity)
    nfeat = cfg["inference"]["orb_nfeatures"]
    thr = cfg["inference"]["orb_match_distance_threshold"]

    ratios = []
    for image_id in tqdm(img_ids, desc="orb", leave=False):
        fname = id2name[image_id]
        clean = np.array(Image.open(clean_dir / fname).convert("RGB"))
        var = np.array(Image.open(var_dir / fname).convert("RGB"))
        ratios.append(orb_match_ratio(clean, var, nfeat, thr))

    tag = variant_tag(variant, dtype, severity)
    return {"task": "orb", "variant": tag,
            "orb_match_ratio": round(float(np.mean(ratios)), 4),
            "n_images": len(ratios)}


# --------------------------------------------------------------------------- #
def evaluate(cfg: Dict, task: str, variant: str, dtype=None, severity=None) -> Dict:
    if task == "orb":
        # clean-vs-clean is trivially 1.0; only meaningful for distorted/enhanced.
        if variant == "clean":
            res = {"task": "orb", "variant": "clean", "orb_match_ratio": 1.0}
        else:
            res = evaluate_orb(cfg, variant, dtype, severity)
    else:
        res = evaluate_coco(cfg, task, variant, dtype, severity)

    tag = variant_tag(variant, dtype, severity)
    out_path = Path(cfg["paths"]["metrics_dir"]) / f"{task}__{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[metrics] {task}/{tag} -> {out_path}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", required=True, choices=["detection", "keypoints", "orb"])
    ap.add_argument("--variant", required=True, choices=["clean", "distorted", "enhanced"])
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--severity", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.variant != "clean" and (not args.dtype or not args.severity):
        ap.error("--dtype and --severity are required for distorted/enhanced variants")
    evaluate(cfg, args.task, args.variant, args.dtype, args.severity)


if __name__ == "__main__":
    main()
