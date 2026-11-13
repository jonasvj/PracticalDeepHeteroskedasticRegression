#!/bin/bash
#SBATCH --job-name=utkface_post_hoc
#SBATCH --array=0-17
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    data=utkface \
    model=resnet \
    method=post_hoc \
    method.calibrate.precompute_features=false \
    method.calibrate.num_epochs=200 \
    method.calibrate.optimizer.lr=1e-4 \
    method.calibrate.optimizer.weight_decay=1e-3 \
    model.feature_extractor.detach_features=true \
    model_name=resnet.pt"

seeds=(1 2 3)
features=(0 1 2 3 4 all)
indices=("[0]" "[1]" "[2]" "[3]" "[4]" "[0,1,2,3,4]")
captures=("[false]" "[false]" "[false]" "[false]" "[true]" "[false,false,false,false,true]")
args_list=()

for seed in "${seeds[@]}"; do
    for i in "${!features[@]}"; do
        args="${base_args} \
            seed=${seed} \
            model.feature_extractor.module_indices_to_extract=${indices[i]} \
            model.feature_extractor.capture_module_input=${captures[i]} \
            model_dir=models/utkface/post_hoc/seed-${seed}/feature-${features[i]} \
            pretrained_model=models/utkface/homo_nll/seed-${seed}/resnet.pt"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
