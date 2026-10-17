#!/bin/bash -l
#SBATCH --job-name=LINGUAL_clean
#SBATCH -p vcggpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=5-00:00:00
#SBATCH --output=%x_%j.out
# usage: sbatch run.sh configs/UDIAT.txt RIPU   (or configs/local/UDIAT.txt)
conda activate graco
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"   # run from this folder
python main.py --cfg "${1:-configs/UDIAT.txt}" --method "${2:-RIPU}"
