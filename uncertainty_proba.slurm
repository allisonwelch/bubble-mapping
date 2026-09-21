#!/bin/bash
#SBATCH --job-name=unc-proba
#SBATCH --partition=t1small
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/amwelch3/bubble-mapping/logs/unc_proba_%j.out
#SBATCH --error=/home/amwelch3/bubble-mapping/logs/unc_proba_%j.err
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load slurm

set +u
eval "$(conda shell.bash hook)"
conda activate venv-bubble
set -u

cd ~/bubble-mapping
mkdir -p logs

RUN_DIR=${RUN_DIR:-/home/amwelch3/bubble-mapping/data/deploy_out/Octopus_10212025_20260915}
DRAWS=${DRAWS:-500}
SEASON=${SEASON:-annual}
SWEEP=${SWEEP:-0.5,0.6,0.7}
OUT_DIR=${OUT_DIR:-$RUN_DIR/uncertainty_proba}

CKPT_DIR=data/models/SWIN/AE/20260428-1537_SWINxAE_continued
for f in "$RUN_DIR/bubbles.gpkg" \
    "$CKPT_DIR/grouper_rf.joblib" \
    "$CKPT_DIR/classifier_rf.joblib" \
    "$CKPT_DIR/artifacts.json"; do
    [[ -e "$f" ]] || { echo "MISSING: $f"; exit 1; }
done

mkdir -p "$OUT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Run dir: $RUN_DIR"
echo "Out dir: $OUT_DIR"
echo "Draws: $DRAWS, sweep=$SWEEP, season=$SEASON"

python -m tools.deploy.uncertainty_proba \
    --bubbles "$RUN_DIR/bubbles.gpkg" \
    --out-dir "$OUT_DIR" \
    --season "$SEASON" \
    --draws "$DRAWS" \
    --threshold-sweep "$SWEEP"

echo "Job finished: $(date)"
