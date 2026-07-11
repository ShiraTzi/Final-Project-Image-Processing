"""Print the README results table (markdown) from results/metrics/comparison.csv.

Formatting rules: 3 decimals, en-dash for missing cells, bold for notable wins
(recovery >= +0.04). Keeps the README table mechanically in sync with the CSV.

Usage:
    .venv/bin/python scripts/readme_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402

BOLD_THR = 0.04
SEV_ORDER = {"low": 0, "med": 1, "high": 2}


def _fmt(v, bold=False, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{v:+.3f}" if sign else f"{v:.3f}"
    s = s.replace("-", "−")
    return f"**{s}**" if bold else s


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    df = pd.read_csv(Path(cfg["paths"]["metrics_dir"]) / "comparison.csv")
    df["order"] = df["severity"].map(SEV_ORDER)
    df = df.sort_values(["task", "distortion", "order"])

    print("| task | distortion | severity | SNR (dB) | clean | distorted | enhanced "
          "| finetuned | finetuned+enh | degradation | recovery (enhance) "
          "| recovery (finetune) | recovery (combined) |")
    print("|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in df.iterrows():
        b_enh = (r["recovery_enhance"] or 0) >= BOLD_THR if pd.notna(r["recovery_enhance"]) else False
        b_ft = (r["recovery_finetune"] or 0) >= BOLD_THR if pd.notna(r["recovery_finetune"]) else False
        b_cmb = (r["recovery_combined"] or 0) >= BOLD_THR if pd.notna(r["recovery_combined"]) else False
        print("| " + " | ".join([
            r["task"], r["distortion"], r["severity"], f"{r['snr_db']:.1f}",
            _fmt(r["clean"]), _fmt(r["distorted"]),
            _fmt(r["enhanced"], bold=b_enh), _fmt(r["finetuned"], bold=b_ft),
            _fmt(r["finetuned_enh"], bold=b_cmb),
            _fmt(r["degradation"], sign=True),
            _fmt(r["recovery_enhance"], bold=b_enh, sign=True),
            _fmt(r["recovery_finetune"], bold=b_ft, sign=True),
            _fmt(r["recovery_combined"], bold=b_cmb, sign=True),
        ]) + " |")

    # helper stats for the prose: share of damage recovered per (distortion, task)
    print("\n<!-- share-of-damage-recovered (recovery / -degradation), for the prose: -->")
    for strat in ("recovery_enhance", "recovery_finetune", "recovery_combined"):
        sub = df[pd.notna(df[strat]) & (df["degradation"] < 0)].copy()
        if sub.empty:
            continue
        sub["share"] = sub[strat] / (-sub["degradation"])
        piv = sub.pivot_table(index=["task"], columns=["distortion", "severity"],
                              values="share").round(2)
        print(f"\n{strat}:\n{piv.to_string()}")


if __name__ == "__main__":
    main()
