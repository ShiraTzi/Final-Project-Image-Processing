"""Download COCO data and build fixed, seeded image-id subsets.

Strategy: download the (small) annotations zip once, then fetch ONLY the
subset images individually via each image's ``coco_url``.  This avoids the
~1GB val2017 zip and the ~18GB train2017 zip while still giving us real COCO
ground truth for evaluation and fine-tuning.

The val subset is the single source of truth reused by every variant.

Usage:
    python -m src.data --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

from src.config import load_config, ensure_dirs

ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
PANOPTIC_ZIP_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for data in r.iter_content(chunk_size=chunk):
                f.write(data)
                bar.update(len(data))
    tmp.rename(dest)


def download_annotations(coco_root: Path) -> None:
    """Download + extract annotations_trainval2017.zip if missing."""
    ann_dir = coco_root / "annotations"
    needed = [
        "instances_val2017.json",
        "person_keypoints_val2017.json",
        "instances_train2017.json",
    ]
    if all((ann_dir / n).exists() for n in needed):
        print("[data] annotations already present")
        return
    zip_path = coco_root / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print("[data] downloading annotations zip (~241MB) ...")
        _download_file(ANN_ZIP_URL, zip_path)
    print("[data] extracting annotations ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(coco_root)


def download_panoptic_annotations(coco_root: Path) -> None:
    """Download + extract panoptic GT (json + per-image PNG masks) for val2017.

    The big zip (~821MB) contains panoptic_{train,val}2017.json plus *nested*
    panoptic_{train,val}2017.zip archives of the PNG masks; we extract the val
    PNGs out of the nested zip."""
    ann_dir = coco_root / "annotations"
    val_json = ann_dir / "panoptic_val2017.json"
    val_png_dir = ann_dir / "panoptic_val2017"
    if val_json.exists() and val_png_dir.is_dir() and any(val_png_dir.iterdir()):
        print("[data] panoptic annotations already present")
        return

    zip_path = coco_root / "panoptic_annotations_trainval2017.zip"
    if not zip_path.exists():
        print("[data] downloading panoptic annotations zip (~821MB) ...")
        _download_file(PANOPTIC_ZIP_URL, zip_path)
    print("[data] extracting panoptic annotations ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(coco_root)
    nested = ann_dir / "panoptic_val2017.zip"
    if nested.exists():
        print("[data] extracting nested panoptic_val2017 PNG masks ...")
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(ann_dir)


def _select_image_ids(ann_path: str, n: int, seed: int,
                      require_annotations: bool = True) -> List[int]:
    """Pick n image-ids deterministically. If require_annotations, only keep
    images that have at least one annotation (so detection GT is non-empty)."""
    from pycocotools.coco import COCO

    coco = COCO(ann_path)
    img_ids = sorted(coco.getImgIds())
    if require_annotations:
        img_ids = [i for i in img_ids if len(coco.getAnnIds(imgIds=i)) > 0]
    rng = random.Random(seed)
    rng.shuffle(img_ids)
    return sorted(img_ids[:n])


def _download_images(ann_path: str, img_ids: List[int], out_dir: Path,
                    workers: int = 16) -> None:
    from pycocotools.coco import COCO

    coco = COCO(ann_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = coco.loadImgs(img_ids)
    todo = [im for im in imgs if not (out_dir / im["file_name"]).exists()]
    if not todo:
        print(f"[data] all {len(imgs)} images already in {out_dir}")
        return
    print(f"[data] downloading {len(todo)} images -> {out_dir}")

    def fetch(im: Dict) -> str:
        dest = out_dir / im["file_name"]
        url = im.get("coco_url") or f"http://images.cocodataset.org/{out_dir.name}/{im['file_name']}"
        _download_file(url, dest)
        return im["file_name"]

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, im): im for im in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="images"):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[data] WARN failed {futs[fut]['file_name']}: {e}")
    if errors:
        print(f"[data] {errors} image(s) failed to download")


def build_subsets(cfg: Dict) -> None:
    coco_root = Path(cfg["paths"]["coco_root"])
    ds = cfg["dataset"]
    seed = cfg["seed"]

    download_annotations(coco_root)
    if "segmentation" in cfg.get("tasks", []):
        download_panoptic_annotations(coco_root)

    # --- val subset (single source of truth) ---
    val_ids = _select_image_ids(ds["ann_instances_val"], ds["val_subset_size"], seed)
    Path(ds["val_subset_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(ds["val_subset_file"], "w") as f:
        json.dump({"image_ids": val_ids, "seed": seed, "split": ds["val_split"]}, f, indent=2)
    print(f"[data] val subset: {len(val_ids)} ids -> {ds['val_subset_file']}")
    _download_images(ds["ann_instances_val"], val_ids,
                     coco_root / ds["val_split"])

    # --- train subset (fine-tuning, real GT). seed+1 to decorrelate from val. ---
    train_ids = _select_image_ids(ds["ann_instances_train"], ds["train_subset_size"], seed + 1)
    with open(ds["train_subset_file"], "w") as f:
        json.dump({"image_ids": train_ids, "seed": seed + 1, "split": ds["train_split"]}, f, indent=2)
    print(f"[data] train subset: {len(train_ids)} ids -> {ds['train_subset_file']}")
    _download_images(ds["ann_instances_train"], train_ids,
                     coco_root / ds["train_split"])


def load_subset_ids(path: str) -> List[int]:
    with open(path) as f:
        return json.load(f)["image_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    build_subsets(cfg)


if __name__ == "__main__":
    main()
