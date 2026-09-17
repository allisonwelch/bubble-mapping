#!/bin/bash
#SBATCH --job-name=mc_flux
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mc_flux_%j.out
#SBATCH --error=logs/mc_flux_%j.err

set -euo pipefail

RUN_DIR=/home/amwelch3/bubble-mapping/data/deploy_out/Octopus_10212025_20260915
PRECISION="${PRECISION:?set PRECISION to precision_lake_ice from bubble_level_summary.csv}"

WORKERS=4
DRAWS_PER_WORKER=125

cd /home/amwelch3/bubble-mapping
OUT_DIR="${RUN_DIR}/mc"
mkdir -p logs "${OUT_DIR}"

export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS ))

python -m tools.deploy.mc_flux \
    --bubbles "${RUN_DIR}/bubbles.gpkg" \
    --out-dir "${OUT_DIR}" \
    --point

PIDS=()
for W in $(seq 0 $(( WORKERS - 1 ))); do
    python -m tools.deploy.mc_flux \
        --bubbles "${RUN_DIR}/bubbles.gpkg" \
        --out-dir "${OUT_DIR}" \
        --draws "${DRAWS_PER_WORKER}" \
        --offset $(( W * DRAWS_PER_WORKER )) \
        --part "${W}" \
        --precision "${PRECISION}" &
    PIDS+=($!)
done

for P in "${PIDS[@]}"; do
    wait "${P}"
done

python -m tools.deploy.mc_flux --summarize "${OUT_DIR}"
