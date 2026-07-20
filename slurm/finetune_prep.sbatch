#!/bin/bash
#SBATCH --job-name=coco_ft_prep
#SBATCH --output=slurm/logs/ft_prep_%j.out
#SBATCH --error=slurm/logs/ft_prep_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=06:00:00

# Prebuild the fine-tuning YOLO dataset (v3 mixture: clean + distorted +
# classically-restored picks).  CPU-heavy because of the BM3D-restored picks
# (~11s/image single-core); the GPU training job then reuses the prebuilt
# dataset via the resume check in build_yolo_dataset.
# Requires the train2017 subset images to be downloaded already (login node:
# `python -m src.data`).
set -euo pipefail
mkdir -p slurm/logs

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
# one BLAS/OpenMP thread per worker process — the fan-out IS the parallelism
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m src.finetune_det --mode build
