#!/bin/bash
#
# Simple SLURM batch script example
#
# --job-name=simple_slurm_job
#SBATCH --output=simple_slurm_job.%j.out
#SBATCH --error=simple_slurm_job.%j.err
# --time=00:10:00
#SBATCH --partition=cpu1T-24h
# --nodes=1
# --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
# Uncomment and set your email if desired:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com
# not use A100 and H200 servers
# #SBATCH --exclude=a100-0-0,a100-0-1,h200-0-0,h200-0-1
echo "Job started: $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Load modules if your cluster uses module system (optional)
# module load python/3.8

# Example: run an inline Python snippet
source /usr/lib/python3.9/site-packages/conda/shell/bin/activate shira_tz
python scripts/run_pipeline.py

echo "Job finished: $(date)"