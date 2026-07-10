"""Evaluation: COCOeval for detection/keypoints + Panoptic FPN PQ for segmentation.

One metric file per (task, variant) under results/metrics/{task}__{tag}.json,
so tables.py / visualize.py can assemble degradation tables and SNR curves.
Segmentation (PQ) is computed by src/segmentation.py running in the dedicated
detectron2 venv; here we just invoke it as a subprocess.

Usage:
    python -m src.metrics --task detection    --variant clean
    python -m src.metrics --task segmentation --variant distorted --dtype motion_blur --severity high
"""
from __future__ import annotations

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import load_config, ensure_dirs, variant_image_dir, variant_tag
from src.data import get_coco, load_subset_ids


# --------------------------------------------------------------------------- #
# COCOeval (detection / keypoints)
# --------------------------------------------------------------------------- #
def _coco_eval(ann_path: str, pred_path: str, img_ids: List[int], iou_type: str):
    from pycocotools.cocoeval import COCOeval

    coco_gt = get_coco(ann_path)
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
# Panoptic segmentation (PQ) — delegated to the detectron2 venv subprocess
# --------------------------------------------------------------------------- #
def evaluate_segmentation(cfg: Dict, variant: str, dtype=None, severity=None) -> Dict:
    """Run src/segmentation.py under the detectron2 venv; it does inference + PQ
    and writes results/metrics/segmentation__{tag}.json, which we then read."""
    py = cfg["segmentation"]["detectron2_python"]
    if not Path(py).is_absolute():
        py = str(Path(cfg["_root"]) / py)
    cmd = [py, "-m", "src.segmentation", "--variant", variant]
    if dtype:
        cmd += ["--dtype", dtype, "--severity", severity]
    print(f"[metrics] segmentation -> subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cfg["_root"])

    tag = variant_tag(variant, dtype, severity)
    out_path = Path(cfg["paths"]["metrics_dir"]) / f"segmentation__{tag}.json"
    with open(out_path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Low-level task: ORB feature matching (no GT needed — clean image is the ref)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _clean_orb_descriptors(path: str, nfeatures: int):
    """Cached (n_keypoints, descriptors) of a clean image — the same clean
    reference is matched against all 19 variants, so extract it once."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"unreadable clean image: {path}")
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, desc = orb.detectAndCompute(img, None)
    return len(kps), desc


def evaluate_features(cfg: Dict, variant: str, dtype=None, severity=None) -> Dict:
    """Mean ORB match ratio over the val subset: descriptors from the clean
    image matched (BFMatcher Hamming, crossCheck) against the variant image;
    good = distance <= threshold; ratio = good / clean keypoints.
    Clean vs clean = 1.0 baseline."""
    import cv2
    from tqdm import tqdm

    nfeatures = int(cfg["orb"]["nfeatures"])
    dist_thr = int(cfg["orb"]["distance_threshold"])

    ds = cfg["dataset"]
    coco = get_coco(ds["ann_instances_val"])
    img_ids = load_subset_ids(ds["val_subset_file"])
    clean_dir = variant_image_dir(cfg, "clean")
    var_dir = variant_image_dir(cfg, variant, dtype, severity)
    if not var_dir.is_dir():
        raise FileNotFoundError(f"variant image dir missing: {var_dir} "
                                "(run the distort/enhance stage first)")

    orb = cv2.ORB_create(nfeatures=nfeatures)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ratios = []
    n_skipped = 0
    for im in tqdm(coco.loadImgs(img_ids), desc="features", leave=False):
        var = cv2.imread(str(var_dir / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if var is None:
            raise FileNotFoundError(f"unreadable variant image: {var_dir / im['file_name']}")
        n_clean_kps, desc1 = _clean_orb_descriptors(str(clean_dir / im["file_name"]), nfeatures)
        if desc1 is None or n_clean_kps == 0:
            n_skipped += 1          # no clean features -> ratio undefined (0/0), skip
            continue
        _, desc2 = orb.detectAndCompute(var, None)
        if desc2 is None:
            ratios.append(0.0)      # variant destroyed all features -> genuine 0
            continue
        matches = matcher.match(desc1, desc2)
        good = [m for m in matches if m.distance <= dist_thr]
        ratios.append(len(good) / n_clean_kps)

    if not ratios:
        raise RuntimeError(f"features/{variant_tag(variant, dtype, severity)}: "
                           "no images could be scored")
    return {
        "task": "features", "variant": variant_tag(variant, dtype, severity),
        "method": "ORB",
        "match_ratio": round(float(np.mean(ratios)), 4),
        "n_images": len(ratios), "n_skipped_no_clean_features": n_skipped,
        "nfeatures": nfeatures, "distance_threshold": dist_thr,
    }


# --------------------------------------------------------------------------- #
def evaluate(cfg: Dict, task: str, variant: str, dtype=None, severity=None) -> Dict:
    if task == "segmentation":
        # subprocess writes the metric json itself; just return it.
        return evaluate_segmentation(cfg, variant, dtype, severity)

    if task == "features":
        res = evaluate_features(cfg, variant, dtype, severity)
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
    ap.add_argument("--task", required=True,
                    choices=["detection", "keypoints", "segmentation", "features"])
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
