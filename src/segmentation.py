"""Panoptic segmentation task: Panoptic FPN (detectron2) + PQ evaluation.

IMPORTANT: this module runs in the dedicated detectron2 venv (`.venv-det`),
which has a detectron2-compatible torch. metrics.py invokes it as a subprocess.
It imports only light helpers from src.config (yaml) — not the rest of the
package — so it does not need the main venv's dependencies.

For a (task=segmentation, variant) it:
  1. runs Panoptic FPN on the fixed val subset in the variant's image dir,
  2. converts detectron2 panoptic output to COCO panoptic PNG + segments_info,
  3. computes PQ/SQ/RQ with panopticapi against the subset panoptic GT,
  4. writes results/metrics/segmentation__{tag}.json.

Usage (under the detectron2 venv):
    .venv-det/bin/python -m src.segmentation --variant clean
    .venv-det/bin/python -m src.segmentation --variant distorted --dtype motion_blur --severity high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import load_config, resolve_image_path, variant_image_dir, variant_tag


def _subset_ids(path: str):
    with open(path) as f:
        return json.load(f)["image_ids"]


def _build_predictor(cfg):
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    name = cfg["segmentation"]["config"]
    dcfg = get_cfg()
    dcfg.merge_from_file(model_zoo.get_config_file(name))
    dcfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)
    dcfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg["segmentation"]["score_threshold"]
    dcfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(dcfg), dcfg


def _panoptic_gt_dir(cfg) -> Path:
    # the per-image GT PNG folder sits next to the panoptic json (same stem).
    return Path(cfg["dataset"]["ann_panoptic_val"]).with_suffix("")


def _prepare_subset_gt(cfg, img_ids):
    """Filter panoptic_val2017.json down to the subset image-ids (once)."""
    out = Path(cfg["paths"]["metrics_dir"]) / "panoptic_gt_subset.json"
    if out.exists():
        return out
    with open(cfg["dataset"]["ann_panoptic_val"]) as f:
        gt = json.load(f)
    idset = set(img_ids)
    sub = {
        "images": [im for im in gt["images"] if im["id"] in idset],
        "annotations": [a for a in gt["annotations"] if a["image_id"] in idset],
        "categories": gt["categories"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sub, f)
    return out


def run(cfg, variant, dtype=None, severity=None) -> dict:
    import cv2
    from detectron2.data import MetadataCatalog
    from panopticapi.utils import id2rgb
    from panopticapi.evaluation import pq_compute

    ds = cfg["dataset"]
    img_ids = _subset_ids(ds["val_subset_file"])
    tag = variant_tag(variant, dtype, severity)

    predictor, dcfg = _build_predictor(cfg)
    meta = MetadataCatalog.get(dcfg.DATASETS.TRAIN[0])
    thing_c2d = {v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()}
    stuff_c2d = {v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()}

    # GT image file names (panoptic json 'images' carry the jpg file_name)
    with open(ds["ann_panoptic_val"]) as f:
        gt = json.load(f)
    id2jpg = {im["id"]: im["file_name"] for im in gt["images"]}  # e.g. 000000.jpg

    img_dir = variant_image_dir(cfg, variant, dtype, severity)
    pred_png_dir = Path(cfg["paths"]["preds_dir"]) / f"segmentation__{tag}"
    pred_png_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    from tqdm import tqdm
    for image_id in tqdm(img_ids, desc=f"panoptic/{tag}", leave=False):
        jpg = id2jpg[image_id]
        img = cv2.imread(str(resolve_image_path(img_dir, jpg)))   # BGR, as detectron2 expects
        if img is None:
            continue
        panoptic_seg, segments_info = predictor(img)["panoptic_seg"]
        pmap = panoptic_seg.cpu().numpy()

        segs = []
        for s in segments_info:
            contig = s["category_id"]
            dataset_id = thing_c2d[contig] if s["isthing"] else stuff_c2d[contig]
            segs.append({"id": int(s["id"]), "category_id": int(dataset_id)})

        png_name = f"{image_id:012d}.png"
        cv2.imwrite(str(pred_png_dir / png_name), cv2.cvtColor(id2rgb(pmap), cv2.COLOR_RGB2BGR))
        annotations.append({"image_id": int(image_id), "file_name": png_name, "segments_info": segs})

    pred_json_path = Path(cfg["paths"]["preds_dir"]) / f"segmentation__{tag}.json"
    with open(pred_json_path, "w") as f:
        json.dump({"annotations": annotations}, f)

    gt_subset = _prepare_subset_gt(cfg, img_ids)
    pq = pq_compute(str(gt_subset), str(pred_json_path),
                    gt_folder=str(_panoptic_gt_dir(cfg)), pred_folder=str(pred_png_dir))

    res = {
        "task": "segmentation", "variant": tag,
        "PQ": round(float(pq["All"]["pq"]), 4),
        "SQ": round(float(pq["All"]["sq"]), 4),
        "RQ": round(float(pq["All"]["rq"]), 4),
        "PQ_things": round(float(pq["Things"]["pq"]), 4),
        "PQ_stuff": round(float(pq["Stuff"]["pq"]), 4),
        "n_images": len(annotations),
    }
    out_path = Path(cfg["paths"]["metrics_dir"]) / f"segmentation__{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[segmentation] {tag}: PQ={res['PQ']} -> {out_path}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", required=True, choices=["clean", "distorted", "enhanced"])
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--severity", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.variant, args.dtype, args.severity)


if __name__ == "__main__":
    main()
