#!/bin/bash
#SBATCH --job-name=qm9_wd_sweep_ensemble
#SBATCH --array=0-29
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    do_calibration=false \
    data=qm9 \
    model=painn \
    method=post_hoc \
    model.variance_head._target_=src.models.painn.PainNNEnsembleVariancePrediction \
    +model.variance_head.ensemble_size=5 \
    model.feature_extractor.detach_features=true \
    model.feature_extractor.module_indices_to_extract=[0,1,2,3,4] \
    model.feature_extractor.capture_module_input=[false,false,false,false,true] \
    model_name=painn.pt"

seeds=(1 2 3)
targets=(7 8)
weight_decays=(1e-4 1e-3 1e-1 1e0 1e1)
args_list=()

for seed in "${seeds[@]}"; do
    for target in "${targets[@]}"; do
        for wd in "${weight_decays[@]}"; do
            pretrained_models="["
            for i in {0..4}; do
                pretrained_models+="models/qm9/wd_sweep/wd-${wd}/target-${target}/post_hoc/seed-${seed}/feature-${i}/painn.pt,"
            done
            pretrained_models="${pretrained_models%,}]"
            args="${base_args} \
                seed=${seed} \
                data.target=${target} \
                method.calibrate.optimizer.weight_decay=${wd} \
                model_dir=models/qm9/wd_sweep/wd-${wd}/target-${target}/post_hoc/seed-${seed}/feature-ensemble \
                pretrained_model=${pretrained_models}"
            args_list+=("${args}")
        done
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
