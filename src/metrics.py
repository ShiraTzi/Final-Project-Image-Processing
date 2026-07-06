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
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import load_config, ensure_dirs, variant_tag
from src.data import load_subset_ids


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
# Panoptic segmentation (PQ) — delegated to the detectron2 venv subprocess
# --------------------------------------------------------------------------- #
def evaluate_segmentation(cfg: Dict, variant: str, dtype=None, severity=None) -> Dict:
    """Run src/segmentation.py under the detectron2 venv; it does inference + PQ
    and writes results/metrics/segmentation__{tag}.json, which we then read."""
    py = os.environ.get("DETECTRON2_PYTHON", cfg["segmentation"]["detectron2_python"])
    if not Path(py).is_absolute():
        py = str(Path(cfg["_root"]) / py)

    py_path = Path(py)
    if not py_path.exists():
        raise FileNotFoundError(
            f"detectron2 interpreter not found: {py_path}. Set configs/config.yaml "
            "segmentation.detectron2_python or the DETECTRON2_PYTHON environment variable."
        )

    probe = subprocess.run(
        [str(py_path), "-c", "import detectron2"],
        cwd=cfg["_root"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "detectron2 is not installed in the configured segmentation environment. "
            f"Interpreter: {py_path}\n"
            f"stderr: {probe.stderr.strip() or '(no stderr)'}\n"
            "Install detectron2 in that environment or point segmentation.detectron2_python "
            "at a working detectron2 venv."
        )

    cmd = [str(py_path), "-m", "src.segmentation", "--variant", variant]
    if dtype:
        cmd += ["--dtype", dtype, "--severity", severity]
    print(f"[metrics] segmentation -> subprocess: {' '.join(cmd)}")
    try:
        completed = subprocess.run(cmd, check=True, cwd=cfg["_root"], capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "segmentation subprocess failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr: {exc.stderr.strip() if exc.stderr else '(no stderr)'}"
        ) from exc
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", flush=True)

    tag = variant_tag(variant, dtype, severity)
    out_path = Path(cfg["paths"]["metrics_dir"]) / f"segmentation__{tag}.json"
    with open(out_path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def evaluate(cfg: Dict, task: str, variant: str, dtype=None, severity=None) -> Dict:
    if task == "segmentation":
        # subprocess writes the metric json itself; just return it.
        return evaluate_segmentation(cfg, variant, dtype, severity)

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
    ap.add_argument("--task", required=True, choices=["detection", "keypoints", "segmentation"])
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
