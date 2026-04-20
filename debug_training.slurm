#!/bin/bash
#SBATCH --job-name=bubble-train
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=185G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate venv-bubble

cd ~/bubble-mapping

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

python training.py

echo "Job finished: $(date)"
