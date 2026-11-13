#!/bin/bash
#SBATCH --job-name=qm9_wd_sweep
#SBATCH --array=0-179
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    data=qm9 \
    model=painn \
    method=post_hoc \
    model.feature_extractor.detach_features=true \
    model_name=painn.pt"

seeds=(1 2 3)
targets=(7 8)
weight_decays=(1e-4 1e-3 1e-1 1e0 1e1)
features=(0 1 2 3 4 all)
indices=("[0]" "[1]" "[2]" "[3]" "[4]" "[0,1,2,3,4]")
captures=("[false]" "[false]" "[false]" "[false]" "[true]" "[false,false,false,false,true]")
args_list=()

for seed in "${seeds[@]}"; do
    for target in "${targets[@]}"; do
        for i in "${!features[@]}"; do
            for wd in "${weight_decays[@]}"; do
                args="${base_args} \
                    seed=${seed} \
                    data.target=${target} \
                    method.calibrate.optimizer.weight_decay=${wd} \
                    model.feature_extractor.module_indices_to_extract=${indices[i]} \
                    model.feature_extractor.capture_module_input=${captures[i]} \
                    model_dir=models/qm9/wd_sweep/wd-${wd}/target-${target}/post_hoc/seed-${seed}/feature-${features[i]} \
                    pretrained_model=models/qm9/target-${target}/homo_nll/seed-${seed}/painn.pt"
                args_list+=("${args}")
            done
        done
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
