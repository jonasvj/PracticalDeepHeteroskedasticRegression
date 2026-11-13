#!/bin/bash
#SBATCH --job-name=utkface_baselines
#SBATCH --array=0-23
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    data=utkface \
    model=resnet \
    method.fit.num_epochs=200 \
    method.fit.optimizer.lr=1e-4 \
    method.fit.optimizer.weight_decay=1e-3 \
    model_name=resnet.pt"

seeds=(1 2 3)
betas=(0.25 0.5 0.75 1)
args_list=()

for method in homo_nll naive_nll natural_nll faithful; do
    method_args=""
    if [ ${method} == "homo_nll" ]; then
        method_args="model.variance_head._target_=src.models.resnet.ResNetConstantVariancePrediction"
    elif [ ${method} == "faithful" ]; then
        method_args="model.feature_extractor.detach_features=true"
    fi
    for seed in "${seeds[@]}"; do
        args="${base_args} \
            seed=${seed} \
            method=${method} \
            ${method_args} \
            model_dir=models/utkface/${method}/seed-${seed}"
        args_list+=("${args}")
    done
done

method="beta_nll"
for seed in "${seeds[@]}"; do
    for beta in "${betas[@]}"; do
        args="${base_args} \
            seed=${seed} \
            method=${method} \
            method.init.beta=${beta} \
            model_dir=models/utkface/${method}/seed-${seed}/beta-${beta}"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
