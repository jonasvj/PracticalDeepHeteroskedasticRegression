#!/bin/bash
#SBATCH --job-name=omol25_homo
#SBATCH --array=0-5
#SBATCH --gres=gpu
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    method=homo_nll \
    model.variance_head._target_=src.models.FeatureModelConstantVariancePrediction \
    ~model.variance_head.feature_keys"

models=(uma allscaip)
seeds=(1 2 3)
args_list=()

for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        args="${base_args} \
            data=${model}_features \
            model=${model} \
            seed=${seed} \
            data.data_split=$((seed - 1)) \
            model_dir=models/${model}/homo_nll/seed-${seed} \
            model_name=${model}.pt"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
