#!/bin/bash
#SBATCH --job-name=bubble-eval
#SBATCH --partition=t1small
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=/home/amwelch3/bubble-mapping/logs/eval_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=END,FAIL

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate venv-bubble

cd ~/bubble-mapping
python evaluation.py