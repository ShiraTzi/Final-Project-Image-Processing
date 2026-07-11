"""Aggregate per-(task,variant) metric JSONs into comparison tables.

Produces:
  results/metrics/summary_long.csv     one row per (task, distortion, severity, variant)
  results/metrics/comparison.csv|md    clean vs distorted vs enhanced (+ degradation/recovery deltas)

Usage:
    python -m src.tables --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import load_config


PRIMARY = {"features": "match_ratio", "detection": "mAP", "keypoints": "mAP",
           "segmentation": "PQ"}


def _mean_snr_lookup(cfg: Dict) -> Dict:
    path = Path(cfg["paths"]["metrics_dir"]) / "snr_index.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.groupby(["distortion", "severity"])["snr_db"].mean().round(3).to_dict()


def _parse_tag(tag: str):
    """clean | <variant>__clean | <variant>__<dtype>__<severity>
    -> (variant, dtype, severity)."""
    if tag == "clean":
        return "clean", None, None
    parts = tag.split("__")
    if len(parts) == 2 and parts[1] == "clean":
        # e.g. finetuned__clean: fine-tuned model evaluated on clean images
        return f"{parts[0]}_clean", None, None
    return parts[0], parts[1], parts[2]


def build_tables(cfg: Dict) -> None:
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    snr = _mean_snr_lookup(cfg)
    rows = []
    for jf in sorted(metrics_dir.glob("*.json")):
        with open(jf) as f:
            res = json.load(f)
        task = res.get("task")
        if task not in PRIMARY:
            continue
        variant, dtype, severity = _parse_tag(res["variant"])
        value = res.get(PRIMARY[task])
        if value is None:
            continue
        rows.append({
            "task": task, "variant": variant, "distortion": dtype, "severity": severity,
            "snr_db": snr.get((dtype, severity)) if dtype else None,
            "metric": PRIMARY[task], "value": value,
        })

    if not rows:
        print("[tables] no metric files found yet")
        return

    long_df = pd.DataFrame(rows).sort_values(["task", "distortion", "severity", "variant"])
    long_path = metrics_dir / "summary_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"[tables] wrote {long_path} ({len(long_df)} rows)")

    # Wide comparison: clean baseline vs distorted vs enhanced (+ finetuned on
    # distorted, and finetuned on enhanced = the two repairs stacked).
    comp_rows = []
    clean_by_task = {r["task"]: r["value"] for r in rows if r["variant"] == "clean"}
    ft_clean = {r["task"]: r["value"] for r in rows if r["variant"] == "finetuned_clean"}
    if ft_clean:
        print(f"[tables] finetuned-on-clean: {ft_clean} "
              f"(pretrained clean: {clean_by_task})")
    cell_df = long_df[~long_df["variant"].isin(["clean", "finetuned_clean"])]
    for (task, dtype, severity), grp in cell_df.groupby(
        ["task", "distortion", "severity"], dropna=False
    ):
        by_variant = grp.set_index("variant")["value"].to_dict()
        clean_v = clean_by_task.get(task)
        dist_v = by_variant.get("distorted")
        enh_v = by_variant.get("enhanced")
        ft_v = by_variant.get("finetuned")
        ft_enh_v = by_variant.get("finetuned_enh")
        comp_rows.append({
            "task": task, "distortion": dtype, "severity": severity,
            "snr_db": grp["snr_db"].iloc[0],
            "clean": clean_v, "distorted": dist_v, "enhanced": enh_v,
            "finetuned": ft_v, "finetuned_enh": ft_enh_v,
            "degradation": _delta(clean_v, dist_v),       # distorted - clean (negative = worse)
            "recovery_enhance": _delta(dist_v, enh_v),    # enhanced - distorted
            "recovery_finetune": _delta(dist_v, ft_v),    # finetuned - distorted
            "recovery_combined": _delta(dist_v, ft_enh_v),  # both repairs - distorted
        })
    comp_df = pd.DataFrame(comp_rows).sort_values(["task", "distortion", "severity"])
    comp_df.to_csv(metrics_dir / "comparison.csv", index=False)
    (metrics_dir / "comparison.md").write_text(comp_df.to_markdown(index=False))
    print(f"[tables] wrote comparison.csv / comparison.md ({len(comp_df)} rows)")


def _delta(a: Optional[float], b: Optional[float]):
    if a is None or b is None:
        return None
    return round(b - a, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    build_tables(load_config(args.config))


if __name__ == "__main__":
    main()
