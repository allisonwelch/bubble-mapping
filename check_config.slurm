#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
#SBATCH --mem=214G

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate venv-bubble

cd ~/bubble-mapping

python checkconfig.py

echo "Job finished: $(date)"

