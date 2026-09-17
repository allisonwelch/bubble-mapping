#!/bin/bash
#SBATCH --job-name=bubble-deploy
#SBATCH --partition=h100
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=185G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=/home/amwelch3/bubble-mapping/logs/deploy_%j.out
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

IMAGE=${IMAGE:-/home/amwelch3/bubble-mapping/data/training_images/AE/Octopus_10212025_thmrgb_duetT_WholeLake_post_cog_rectified.tif}
LAKE=${LAKE:-/home/amwelch3/bubble-mapping/data/geopackages/digitized_shoreline_Octopus_rough.gpkg}
DEPLOY_ROOT=${DEPLOY_ROOT:-/home/amwelch3/bubble-mapping/data/deploy_out}
IMAGE_STEM=$(basename "$IMAGE")
IMAGE_STEM=${IMAGE_STEM%.*}
LAKE_TAG=$(printf '%s' "$IMAGE_STEM" | sed -E 's/^([^_]+_[0-9]{6,8}).*/\1/')
RUN_NAME=${RUN_NAME:-${LAKE_TAG}_$(date +%Y%m%d)}
OUT_DIR=${OUT_DIR:-$DEPLOY_ROOT/$RUN_NAME}
EXTRA=${EXTRA:-}

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