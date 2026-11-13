#!/bin/bash
#SBATCH --job-name=omol25_create_features
#SBATCH --array=0-1
#SBATCH --gres=gpu
#SBATCH --mem=128gb
#SBATCH --cpus-per-task=4

modules=(src.data.create_uma src.data.create_allscaip)

echo Module: ${modules[${SLURM_ARRAY_TASK_ID}]}
python3 -m ${modules[${SLURM_ARRAY_TASK_ID}]}
