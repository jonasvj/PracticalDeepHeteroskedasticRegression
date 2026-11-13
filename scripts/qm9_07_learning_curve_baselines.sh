#!/bin/bash
#SBATCH --job-name=qm9_learning_curve_baselines
#SBATCH --array=0-191
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    data=qm9 \
    model=painn \
    model_name=painn.pt"

seeds=(1 2 3)
target=8
cal_subset_sizes=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 1)
betas=(0.25 0.5 0.75 1)
args_list=()

for method in homo_nll naive_nll natural_nll faithful; do
    method_args=""
    if [ ${method} == "homo_nll" ]; then
        method_args="model.variance_head._target_=src.models.painn.PaiNNConstantVariancePrediction"
    elif [ ${method} == "faithful" ]; then
        method_args="model.feature_extractor.detach_features=true"
    fi
    for seed in "${seeds[@]}"; do
        for subset in "${cal_subset_sizes[@]}"; do
            args="${base_args} \
                seed=${seed} \
                data.target=${target} \
                data.cal_subset_size=${subset} \
                data.cal_subset_seed=${seed} \
                method=${method} \
                ${method_args} \
                model_dir=models/qm9/learning_curve/subset-${subset}/target-${target}/${method}/seed-${seed} \
                pretrained_model=models/qm9/target-${target}/${method}/seed-${seed}/painn.pt"
            args_list+=("${args}")
        done
    done
done

method="beta_nll"
for seed in "${seeds[@]}"; do
    for beta in "${betas[@]}"; do
        for subset in "${cal_subset_sizes[@]}"; do
            args="${base_args} \
                seed=${seed} \
                data.target=${target} \
                data.cal_subset_size=${subset} \
                data.cal_subset_seed=${seed} \
                method=${method} \
                method.init.beta=${beta} \
                model_dir=models/qm9/learning_curve/subset-${subset}/target-${target}/${method}/seed-${seed}/beta-${beta} \
                pretrained_model=models/qm9/target-${target}/${method}/seed-${seed}/beta-${beta}/painn.pt"
            args_list+=("${args}")
        done
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
