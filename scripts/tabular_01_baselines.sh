#!/bin/bash
#SBATCH --job-name=tabular_baselines
#SBATCH --array=0-79
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    model=ft_transformer \
    method.fit.num_epochs=10000 \
    method.fit.patience=17 \
    method.fit.min_epochs=100 \
    method.fit.es_criterion=rmse \
    method.fit.optimizer.lr=1e-4 \
    method.fit.optimizer.weight_decay=1e-5 \
    method.fit.scheduler._target_=torch.optim.lr_scheduler.ConstantLR \
    +method.fit.scheduler.factor=1 \
    ~method.fit.scheduler.eta_min \
    ~method.fit.scheduler.T_max \
    model_name=ft_transformer.pt"

datasets=(california_housing year)
declare -A d_numerical=([california_housing]=8 [year]=90)
seeds=(1 2 3 4 5)
betas=(0.25 0.5 0.75 1)
args_list=()

for dataset in "${datasets[@]}"; do
    data_args="data=${dataset} model.backbone.d_numerical=${d_numerical[${dataset}]}"

    for method in homo_nll naive_nll natural_nll faithful; do
        method_args=""
        if [ ${method} == "homo_nll" ]; then
            method_args="model.variance_head._target_=src.models.ft_transformer.FTTransformerConstantVariancePrediction"
        elif [ ${method} == "faithful" ]; then
            method_args="model.feature_extractor.detach_features=true"
        fi
        for seed in "${seeds[@]}"; do
            args="${base_args} \
                ${data_args} \
                seed=${seed} \
                method=${method} \
                ${method_args} \
                model_dir=models/${dataset}/${method}/seed-${seed}"
            args_list+=("${args}")
        done
    done

    method="beta_nll"
    for seed in "${seeds[@]}"; do
        for beta in "${betas[@]}"; do
            args="${base_args} \
                ${data_args} \
                seed=${seed} \
                method=${method} \
                method.init.beta=${beta} \
                model_dir=models/${dataset}/${method}/seed-${seed}/beta-${beta}"
            args_list+=("${args}")
        done
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
