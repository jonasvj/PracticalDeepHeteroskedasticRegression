#!/bin/bash
#SBATCH --job-name=tabular_post_hoc
#SBATCH --array=0-49
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    model=ft_transformer \
    method=post_hoc \
    method.calibrate.precompute_features=false \
    method.calibrate.num_epochs=200 \
    method.calibrate.optimizer.lr=1e-4 \
    method.calibrate.optimizer.weight_decay=1e-5 \
    model.feature_extractor.detach_features=true \
    model_name=ft_transformer.pt"

datasets=(california_housing year)
declare -A d_numerical=([california_housing]=8 [year]=90)
seeds=(1 2 3 4 5)
features=(0 1 2 3 all)
indices=("[0]" "[1]" "[2]" "[3]" "[0,1,2,3]")
captures=("[false]" "[false]" "[false]" "[true]" "[false,false,false,true]")
args_list=()

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        for i in "${!features[@]}"; do
            args="${base_args} \
                data=${dataset} \
                model.backbone.d_numerical=${d_numerical[${dataset}]} \
                seed=${seed} \
                model.feature_extractor.module_indices_to_extract=${indices[i]} \
                model.feature_extractor.capture_module_input=${captures[i]} \
                model_dir=models/${dataset}/post_hoc/seed-${seed}/feature-${features[i]} \
                pretrained_model=models/${dataset}/homo_nll/seed-${seed}/ft_transformer.pt"
            args_list+=("${args}")
        done
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
