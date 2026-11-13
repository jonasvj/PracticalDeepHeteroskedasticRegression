#!/bin/bash
#SBATCH --job-name=omol25_learning_curve_ensemble
#SBATCH --array=0-23
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    do_calibration=false \
    use_var_scale_during_normal_test=true \
    data=uma_features \
    model=uma \
    method=post_hoc \
    model.variance_head._target_=src.models.FeatureModelEnsembleVariancePrediction \
    model_name=uma.pt"

features=(
    module.backbone.blocks.0
    module.backbone.blocks.1
    module.backbone.blocks.2
    module.backbone.blocks.3
    module.output_heads.energyandforcehead.head.energy_block.0
    module.output_heads.energyandforcehead.head.energy_block.2
    module.output_heads.energyandforcehead.head.energy_block.4
)
all_features=$(IFS=,; echo "[${features[*]}]")
seeds=(1 2 3)
cal_subset_sizes=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 1)
args_list=()

for seed in "${seeds[@]}"; do
    for subset in "${cal_subset_sizes[@]}"; do
        pretrained_models="["
        for i in "${!features[@]}"; do
            pretrained_models+="models/uma/learning_curve/subset-${subset}/post_hoc/seed-${seed}/feature-${i}/uma.pt,"
        done
        pretrained_models="${pretrained_models%,}]"
        args="${base_args} \
            seed=${seed} \
            data.data_split=$((seed - 1)) \
            data.cal_subset_size=${subset} \
            model.variance_head.feature_keys=${all_features} \
            model_dir=models/uma/learning_curve/subset-${subset}/post_hoc/seed-${seed}/feature-ensemble \
            pretrained_model=${pretrained_models}"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
