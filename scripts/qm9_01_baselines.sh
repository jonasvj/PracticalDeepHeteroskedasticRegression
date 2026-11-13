#!/bin/bash
#SBATCH --job-name=qm9_baselines
#SBATCH --array=0-239
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    data=qm9 \
    model=painn \
    model_name=painn.pt"

seeds=(1 2 3)
targets=(1 2 3 4 6 7 8 9 10 11)
betas=(0.25 0.5 0.75 1)
args_list=()

method="homo_nll"
for seed in "${seeds[@]}"; do
    for target in "${targets[@]}"; do
        args="${base_args} \
            seed=${seed} \
            data.target=${target} \
            method=${method} \
            model.variance_head._target_=src.models.painn.PaiNNConstantVariancePrediction \
            model_dir=models/qm9/target-${target}/${method}/seed-${seed}"
        args_list+=("${args}")
    done
done

method="naive_nll"
for seed in "${seeds[@]}"; do
    for target in "${targets[@]}"; do
        args="${base_args} \
            seed=${seed} \
            data.target=${target} \
            method=${method} \
            model_dir=models/qm9/target-${target}/${method}/seed-${seed}"
        args_list+=("${args}")
    done
done

method="natural_nll"
for seed in "${seeds[@]}"; do
    for target in "${targets[@]}"; do
        normalize_loss=false
        if [ ${target} -ge 7 ] && [ ${target} -le 10 ]; then
            normalize_loss=true
        fi
        args="${base_args} \
            seed=${seed} \
            data.target=${target} \
            method=${method} \
            method.init.normalize_loss=${normalize_loss} \
            model_dir=models/qm9/target-${target}/${method}/seed-${seed}"
        args_list+=("${args}")
    done
done

method="beta_nll"
for seed in "${seeds[@]}"; do
    for target in "${targets[@]}"; do
        for beta in "${betas[@]}"; do
            args="${base_args} \
                seed=${seed} \
                data.target=${target} \
                method=${method} \
                method.init.beta=${beta} \
                model_dir=models/qm9/target-${target}/${method}/seed-${seed}/beta-${beta}"
            args_list+=("${args}")
        done
    done
done

method="faithful"
for seed in "${seeds[@]}"; do
    for target in "${targets[@]}"; do
        args="${base_args} \
            seed=${seed} \
            data.target=${target} \
            method=${method} \
            model.feature_extractor.detach_features=true \
            model_dir=models/qm9/target-${target}/${method}/seed-${seed}"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
