#!/bin/bash
#SBATCH --job-name=qm9_learning_curve_post_hoc
#SBATCH --array=0-143
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    data=qm9 \
    model=painn \
    method=post_hoc \
    method.calibrate._target_=src.methods.PostHocRegression.calibrate_with_earlystopping \
    method.calibrate.num_epochs=5000 \
    method.calibrate.patience=50 \
    method.calibrate.min_epochs=100 \
    method.calibrate.optimizer.weight_decay=0 \
    +method.calibrate.train_val_split_ratio=0.8 \
    model.feature_extractor.detach_features=true \
    model_name=painn.pt"

seeds=(1 2 3)
target=8
cal_subset_sizes=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 1)
features=(0 1 2 3 4 all)
indices=("[0]" "[1]" "[2]" "[3]" "[4]" "[0,1,2,3,4]")
captures=("[false]" "[false]" "[false]" "[false]" "[true]" "[false,false,false,false,true]")
args_list=()

for seed in "${seeds[@]}"; do
    for i in "${!features[@]}"; do
        for subset in "${cal_subset_sizes[@]}"; do
            args="${base_args} \
                seed=${seed} \
                data.target=${target} \
                data.cal_subset_size=${subset} \
                data.cal_subset_seed=${seed} \
                +method.calibrate.split_seed=${seed} \
                model.feature_extractor.module_indices_to_extract=${indices[i]} \
                model.feature_extractor.capture_module_input=${captures[i]} \
                model_dir=models/qm9/learning_curve/subset-${subset}/target-${target}/post_hoc/seed-${seed}/feature-${features[i]} \
                pretrained_model=models/qm9/target-${target}/homo_nll/seed-${seed}/painn.pt"
            args_list+=("${args}")
        done
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
