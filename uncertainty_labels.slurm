#!/bin/bash
#SBATCH --job-name=unc-labels
#SBATCH --partition=t1small
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/home/amwelch3/bubble-mapping/logs/unc_labels_%j.out
#SBATCH --error=/home/amwelch3/bubble-mapping/logs/unc_labels_%j.err
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
WORKERS=${WORKERS:-8}
SEASON=${SEASON:-annual}
TERMS=${TERMS:-grouper,classifier}
OUT_DIR=${OUT_DIR:-$RUN_DIR/uncertainty_labels}

CKPT_DIR=data/models/SWIN/AE/20260428-1537_SWINxAE_continued
for f in "$RUN_DIR/bubbles.gpkg" \
    "$CKPT_DIR/grouper_rf.joblib" \
    "$CKPT_DIR/classifier_rf.joblib" \
    "$CKPT_DIR/artifacts.json"; do
    [[ -e "$f" ]] || { echo "MISSING: $f"; exit 1; }
done

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/draws_*.csv "$OUT_DIR"/meta_*.json

OMP=$(( SLURM_CPUS_PER_TASK / WORKERS ))
[[ "$OMP" -ge 1 ]] || OMP=1
export OMP_NUM_THREADS=$OMP
export OPENBLAS_NUM_THREADS=$OMP
export MKL_NUM_THREADS=$OMP

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Run dir: $RUN_DIR"
echo "Out dir: $OUT_DIR"
echo "Draws: $DRAWS across $WORKERS workers, terms=$TERMS, season=$SEASON"

python -m tools.deploy.uncertainty_labels \
    --bubbles "$RUN_DIR/bubbles.gpkg" \
    --out-dir "$OUT_DIR" \
    --season "$SEASON" \
    --point

PIDS=()
for W in $(seq 0 $(( WORKERS - 1 ))); do
    python -m tools.deploy.uncertainty_labels \
        --bubbles "$RUN_DIR/bubbles.gpkg" \
        --out-dir "$OUT_DIR" \
        --season "$SEASON" \
        --terms "$TERMS" \
        --draws "$DRAWS" \
        --shard "$W/$WORKERS" &
    PIDS+=($!)
done

for P in "${PIDS[@]}"; do
    wait "$P"
done

python -m tools.deploy.uncertainty_labels --summarize "$OUT_DIR"

echo "Job finished: $(date)"
