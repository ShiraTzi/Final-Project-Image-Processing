#!/bin/bash
#SBATCH --job-name=coco_finetune
#SBATCH --output=slurm/logs/finetune_%j.out
#SBATCH --error=slurm/logs/finetune_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Fine-tune Faster R-CNN on distorted train2017, then evaluate on distorted val.
set -euo pipefail
mkdir -p slurm/logs

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -m src.finetune_det --mode both "$@"
