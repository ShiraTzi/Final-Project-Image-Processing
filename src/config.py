"""Config loading + shared path/variant helpers.

Single source of truth: everything reads ``configs/config.yaml`` so that the
clean / distorted / enhanced / finetuned variants stay directly comparable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Repo root = parent of the src/ directory holding this file.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML config and resolve every path under ``paths`` to absolute."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the repo root for robustness to CWD.
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str((ROOT / val).resolve()) if not os.path.isabs(val) else val
    for key, val in cfg.get("dataset", {}).items():
        if isinstance(val, str) and val.endswith((".json",)):
            cfg["dataset"][key] = str((ROOT / val).resolve()) if not os.path.isabs(val) else val

    cfg["_root"] = str(ROOT)
    return cfg


def variant_image_dir(cfg: Dict[str, Any], variant: str, dtype: str | None = None,
                       severity: str | None = None) -> Path:
    """Return the image directory for a given variant.

    variant ∈ {"clean", "distorted", "enhanced"}.
    For distorted/enhanced, ``dtype`` and ``severity`` are required.
    """
    if variant == "clean":
        return Path(cfg["paths"]["coco_root"]) / cfg["dataset"]["val_split"]
    if variant in ("distorted", "finetuned"):
        # "finetuned" = fine-tuned detector evaluated on the distorted images.
        return Path(cfg["paths"]["distorted_root"]) / dtype / severity
    if variant == "enhanced":
        return Path(cfg["paths"]["enhanced_root"]) / dtype / severity
    raise ValueError(f"unknown variant: {variant}")


def variant_tag(variant: str, dtype: str | None = None, severity: str | None = None) -> str:
    """Stable short tag used in prediction/metric filenames."""
    if variant == "clean":
        return "clean"
    return f"{variant}__{dtype}__{severity}"


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create all output directories declared in the config."""
    for key in ("preds_dir", "metrics_dir", "figures_dir", "models_dir", "splits_dir",
                "distorted_root", "enhanced_root"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
