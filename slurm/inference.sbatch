#!/bin/bash
#SBATCH --job-name=coco_infer
#SBATCH --output=slurm/logs/infer_%j.out
#SBATCH --error=slurm/logs/infer_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# GPU inference + evaluation only (assumes data/distortions/enhancements exist).
set -euo pipefail
mkdir -p slurm/logs

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/run_pipeline.py --only infer eval report "$@"
