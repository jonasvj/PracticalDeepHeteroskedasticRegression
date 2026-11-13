#!/bin/bash
#SBATCH --job-name=tabular_post_hoc_ensemble
#SBATCH --array=0-9
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    do_calibration=false \
    model=ft_transformer \
    method=post_hoc \
    model.variance_head._target_=src.models.ft_transformer.FTTransformerEnsembleVariancePrediction \
    +model.variance_head.ensemble_size=4 \
    model.feature_extractor.detach_features=true \
    model.feature_extractor.module_indices_to_extract=[0,1,2,3] \
    model.feature_extractor.capture_module_input=[false,false,false,true] \
    model_name=ft_transformer.pt"

datasets=(california_housing year)
declare -A d_numerical=([california_housing]=8 [year]=90)
seeds=(1 2 3 4 5)
args_list=()

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        pretrained_models="["
        for i in {0..3}; do
            pretrained_models+="models/${dataset}/post_hoc/seed-${seed}/feature-${i}/ft_transformer.pt,"
        done
        pretrained_models="${pretrained_models%,}]"
        args="${base_args} \
            data=${dataset} \
            model.backbone.d_numerical=${d_numerical[${dataset}]} \
            seed=${seed} \
            model_dir=models/${dataset}/post_hoc/seed-${seed}/feature-ensemble \
            pretrained_model=${pretrained_models}"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
