#!/bin/bash
#SBATCH --job-name=bubble-deploy
#SBATCH --partition=l40s
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=185G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=/home/amwelch3/bubble-mapping/logs/deploy_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Whole-lake deployment, start to finish: ortho -> bubbles -> seeps -> flux.
# The GPU is only needed for the detect stage, but postproc takes about a
# minute, so the whole chain runs in one job and you get the flux table out
# of the log.
#
# Submit from chinookgpu.alaska.edu:
#   IMAGE=data/training_images/AE/<ortho>.tif \
#   LAKE=data/geopackages/<lake>.gpkg \
#   OUT_DIR=data/deploy_out/<lake>_<YYYYMMDD> \
#   sbatch deploy_lake.slurm
#
# WALL TIME. Cost is (tiles) x (patches per tile) x (MC samples). Run once with
# EXTRA="--limit-tiles 20" to measure the per-tile rate, then set --time from
# that. l40s is the right partition for inference; keep h100 for training.
#
# LAKE POLYGON. Not optional in practice -- without it the detector runs over
# shore and snow-covered bank, whose detections are false positives by
# construction and do not all land in the lowest-rate class.

# Fail the job on the first error, not silently: the last command here is an
# echo, so without this a crashed python still exits 0, slurm records the job
# as COMPLETED and --mail-type=FAIL never fires.
set -euo pipefail

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate venv-bubble

cd ~/bubble-mapping
mkdir -p logs

IMAGE=${IMAGE:?set IMAGE=/home/amwelch3/bubble-mapping/data/training_images/AE/Octopus_10212025_thmrgb_duetT_WholeLake_post_cog_rectified.tif}
LAKE=${LAKE:?set LAKE=/home/amwelch3/bubble-mapping/data/geopackages/digitized_shoreline_Octopus_rough.gpkg}
OUT_DIR=${OUT_DIR:?set OUT_DIR=/home/amwelch3/bubble-mapping/data/deploy_out/Octopus_10212025_20260914/}
EXTRA=${EXTRA:-}

# Everything below lives under the gitignored data/, so none of it arrives with
# a git pull. Fail here rather than hours into the GPU stage.
#
# NO LABELER GEOPACKAGES ARE NEEDED. Since 2026-09-11 the grouper and the
# classifier are frozen joblibs built by `python -m tools.deploy.build_artifacts`
# and they travel with the checkpoint. This list is the complete set of
# non-git inputs.
CKPT_DIR=data/models/SWIN/AE/20260428-1537_SWINxAE_continued
for f in "$IMAGE" "$LAKE" \
    "$CKPT_DIR/20260428-1537_SWINxAE.weights.pt" \
    "$CKPT_DIR/grouper_rf.joblib" \
    "$CKPT_DIR/classifier_rf.joblib" \
    "$CKPT_DIR/artifacts.json"; do
    [[ -e "$f" ]] || { echo "MISSING: $f"; exit 1; }
done

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

python deploy.py \
    --image "$IMAGE" \
    --lake "$LAKE" \
    --out-dir "$OUT_DIR" \
    $EXTRA

echo "Job finished: $(date)"