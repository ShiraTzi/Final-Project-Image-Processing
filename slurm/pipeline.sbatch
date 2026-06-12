#!/bin/bash
#SBATCH --job-name=coco_robust
#SBATCH --output=slurm/logs/pipeline_%j.out
#SBATCH --error=slurm/logs/pipeline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Full benchmark end-to-end on one GPU node.
set -euo pipefail
mkdir -p slurm/logs

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/run_pipeline.py "$@"
