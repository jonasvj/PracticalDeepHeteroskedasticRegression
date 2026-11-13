#!/bin/bash
#SBATCH --job-name=omol25_learning_curve
#SBATCH --array=0-215
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    data=uma_features \
    model=uma \
    model_name=uma.pt"

post_hoc_args="\
    method=post_hoc \
    method.calibrate._target_=src.methods.PostHocRegression.calibrate_with_earlystopping \
    method.calibrate.num_epochs=5000 \
    method.calibrate.patience=50 \
    method.calibrate.min_epochs=100 \
    method.calibrate.optimizer.weight_decay=0 \
    method.calibrate.fit_var_scale_factor=true \
    +method.calibrate.train_val_split_ratio=0.8"

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
        args="${base_args} \
            seed=${seed} \
            data.data_split=$((seed - 1)) \
            data.cal_subset_size=${subset} \
            method=homo_nll \
            model.variance_head._target_=src.models.FeatureModelConstantVariancePrediction \
            ~model.variance_head.feature_keys \
            model_dir=models/uma/learning_curve/subset-${subset}/homo_nll/seed-${seed}"
        args_list+=("${args}")
    done
done

for seed in "${seeds[@]}"; do
    for subset in "${cal_subset_sizes[@]}"; do
        for i in "${!features[@]}"; do
            args="${base_args} \
                ${post_hoc_args} \
                seed=${seed} \
                data.data_split=$((seed - 1)) \
                data.cal_subset_size=${subset} \
                +method.calibrate.split_seed=${seed} \
                model.variance_head.feature_keys=[${features[i]}] \
                model_dir=models/uma/learning_curve/subset-${subset}/post_hoc/seed-${seed}/feature-${i}"
            args_list+=("${args}")
        done

        args="${base_args} \
            ${post_hoc_args} \
            seed=${seed} \
            data.data_split=$((seed - 1)) \
            data.cal_subset_size=${subset} \
            +method.calibrate.split_seed=${seed} \
            model.variance_head.feature_keys=${all_features} \
            model_dir=models/uma/learning_curve/subset-${subset}/post_hoc/seed-${seed}/feature-all"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
