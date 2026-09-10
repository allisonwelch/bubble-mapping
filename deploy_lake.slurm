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
#   LAKE=data/training/AE/lakes.gpkg:<layer> \
#   OUT_DIR=data/results/SWIN/AE/deploy/<lake>_<date> \
#   sbatch deploy_lake.slurm
#
# WALL TIME. Cost is (tiles) x (patches per tile) x (MC samples). Run once with
# EXTRA="--limit-tiles 20" to measure the per-tile rate, then set --time from
# that. l40s is the right partition for inference; keep h100 for training.
#
# LAKE POLYGON. Not optional in practice -- without it the detector runs over
# shore and snow-covered bank, whose detections are false positives by
# construction and do not all land in the lowest-rate class.

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate bubble-mapping

cd ~/bubble-mapping

IMAGE=${IMAGE:?set IMAGE=path/to/whole_lake_ortho.tif}
LAKE=${LAKE:?set LAKE=path/to/lake_polygon.gpkg[:layer]}
OUT_DIR=${OUT_DIR:?set OUT_DIR=data/results/SWIN/AE/deploy/<lake>_<date>}
EXTRA=${EXTRA:-}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

python deploy.py \
    --image "$IMAGE" \
    --lake "$LAKE" \
    --out-dir "$OUT_DIR" \
    $EXTRA

echo "Job finished: $(date)"