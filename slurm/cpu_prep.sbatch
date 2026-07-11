#!/bin/bash
#SBATCH --job-name=coco_prep
#SBATCH --output=slurm/logs/prep_%j.out
#SBATCH --error=slurm/logs/prep_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00

# CPU-only stages (distort + enhance) — no GPU needed; lets the GPU job start
# straight at inference.
set -euo pipefail
mkdir -p slurm/logs

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/run_pipeline.py --only distort enhance
