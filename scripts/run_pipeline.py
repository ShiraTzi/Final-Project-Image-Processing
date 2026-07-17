"""Orchestrate the full benchmark: data -> distort -> enhance -> infer -> eval
-> (fine-tune) -> tables -> figures.

Resumable: every step skips work whose output JSON already exists (unless
--force). Heavy GPU steps (detection/keypoints inference, fine-tuning) are the
ones to run under SLURM; ORB/distortion/enhancement are CPU.

Usage:
    python scripts/run_pipeline.py                 # everything
    python scripts/run_pipeline.py --skip-finetune
    python scripts/run_pipeline.py --only infer eval
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# allow running as a script (python scripts/run_pipeline.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ensure_dirs, variant_tag  # noqa: E402

# Per-variant failures collected across stages: the run keeps going (one bad
# variant shouldn't kill a long GPU job) but finishes with a summary + exit 1
# so SLURM/batch wrappers don't report an incomplete run as success.
FAILURES: list = []


def _exists(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"[skip] {path.name} exists")
        return True
    return False


def stage_data(cfg, force):
    from src.data import build_subsets
    if _exists(Path(cfg["dataset"]["val_subset_file"]), force):
        return
    build_subsets(cfg)


def stage_distort(cfg, force):
    from src.distortions import build_distorted_sets
    if _exists(Path(cfg["paths"]["metrics_dir"]) / "snr_index.csv", force):
        return
    build_distorted_sets(cfg)


def stage_enhance(cfg, force):
    from src.enhancement import build_enhanced_sets
    workers = int(cfg.get("enhancement", {}).get("workers", 8))
    build_enhanced_sets(cfg, workers=workers)   # internally skips existing images


def _variants(cfg):
    """Yield (variant, dtype, severity) over clean + every distorted/enhanced cell."""
    yield "clean", None, None
    for dtype, spec in cfg["distortions"].items():
        for severity in spec["severities"]:
            yield "distorted", dtype, severity
            yield "enhanced", dtype, severity


def stage_infer(cfg, force):
    from src.inference import run_inference
    coco_tasks = [t for t in cfg["tasks"] if t in ("detection", "keypoints")]
    for variant, dtype, severity in _variants(cfg):
        for task in coco_tasks:
            tag = variant_tag(variant, dtype, severity)
            out = Path(cfg["paths"]["preds_dir"]) / f"{task}__{tag}.json"
            if _exists(out, force):
                continue
            try:
                run_inference(cfg, task, variant, dtype, severity)
            except Exception:  # noqa: BLE001 — keep the long run going
                traceback.print_exc()
                print(f"[infer] FAILED {task}/{tag}")
                FAILURES.append(f"infer:{task}/{tag}")


def stage_eval(cfg, force):
    from src.metrics import evaluate
    for variant, dtype, severity in _variants(cfg):
        for task in cfg["tasks"]:
            tag = variant_tag(variant, dtype, severity)
            out = Path(cfg["paths"]["metrics_dir"]) / f"{task}__{tag}.json"
            if _exists(out, force):
                continue
            try:
                evaluate(cfg, task, variant, dtype, severity)
            except Exception:  # noqa: BLE001 — one bad variant shouldn't abort the run
                traceback.print_exc()
                print(f"[eval] FAILED {task}/{tag}")
                FAILURES.append(f"eval:{task}/{tag}")


def stage_finetune(cfg, force):
    from src.finetune_det import train, evaluate_finetuned
    ckpt = Path(cfg["finetune"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = Path(cfg["_root"]) / ckpt
    if not _exists(ckpt, force):
        train(cfg)
    FAILURES.extend(evaluate_finetuned(cfg, force))


def stage_report(cfg, force):
    from src.tables import build_tables
    from src.visualize import (
        plot_acc_vs_snr, plot_recovery_bars, plot_per_class_ap,
        plot_per_class_comparison, plot_image_grids, plot_annotated_grids,
        plot_orb_match_grids, plot_panoptic_grids,
    )
    build_tables(cfg)
    plot_acc_vs_snr(cfg)
    plot_recovery_bars(cfg)
    plot_per_class_ap(cfg)
    plot_per_class_comparison(cfg)
    plot_image_grids(cfg)
    plot_annotated_grids(cfg)
    plot_orb_match_grids(cfg)
    plot_panoptic_grids(cfg)


STAGES = {
    "data": stage_data, "distort": stage_distort, "enhance": stage_enhance,
    "infer": stage_infer, "eval": stage_eval, "finetune": stage_finetune,
    "report": stage_report,
}
DEFAULT_ORDER = ["data", "distort", "enhance", "infer", "eval", "finetune", "report"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    ap.add_argument("--only", nargs="+", choices=list(STAGES), help="run only these stages")
    ap.add_argument("--skip-finetune", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    order = args.only if args.only else DEFAULT_ORDER
    for name in order:
        if name == "finetune" and args.skip_finetune:
            print("[pipeline] skipping finetune")
            continue
        print(f"\n=== stage: {name} ===")
        STAGES[name](cfg, args.force)

    if FAILURES:
        print(f"\n[pipeline] {len(FAILURES)} step(s) FAILED — report artifacts "
              "may be incomplete:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
