#!/bin/bash
#SBATCH --job-name=coco_prep
#SBATCH --output=slurm/logs/prep_%j.out
#SBATCH --error=slurm/logs/prep_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --nodelist=rxn01
#SBATCH --cpus-per-task=128
#SBATCH --mem=180G
#SBATCH --time=12:00:00

# CPU-only stages (distort + enhance) — no GPU needed; lets the GPU job start
# straight at inference.  BM3D costs ~2-4 core-minutes/image and is memory-
# bandwidth-bound, so run on the 128-core rxn nodes (more memory channels);
# the enhance stage fans out one process per image.
set -euo pipefail
mkdir -p slurm/logs

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
# one BLAS/OpenMP thread per worker process — the fan-out IS the parallelism
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/run_pipeline.py --only distort enhance
