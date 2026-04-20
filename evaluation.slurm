#!/bin/bash
#SBATCH --job-name=bubble-eval
#SBATCH --partition=l40s
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=185G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=logs/eval_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=END,FAIL

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate venv-bubble

cd ~/bubble-mapping
python evaluation.py