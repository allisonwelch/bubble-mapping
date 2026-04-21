#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate venv-bubble

cd ~/bubble-mapping

python checkconfig.py

echo "Job finished: $(date)"

