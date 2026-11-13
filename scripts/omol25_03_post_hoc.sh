#!/bin/bash
#SBATCH --job-name=omol25_post_hoc
#SBATCH --array=0-50
#SBATCH --gres=gpu
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    method=post_hoc \
    method.calibrate.fit_var_scale_factor=true"

uma_features=(
    module.backbone.blocks.0
    module.backbone.blocks.1
    module.backbone.blocks.2
    module.backbone.blocks.3
    module.output_heads.energyandforcehead.head.energy_block.0
    module.output_heads.energyandforcehead.head.energy_block.2
    module.output_heads.energyandforcehead.head.energy_block.4
)
allscaip_features=(
    module.backbone.transformer_blocks.0.node_ffn.ffn_norm
    module.backbone.transformer_blocks.1.node_ffn.ffn_norm
    module.backbone.transformer_blocks.2.node_ffn.ffn_norm
    module.backbone.transformer_blocks.3.node_ffn.ffn_norm
    module.backbone.transformer_blocks.4.node_ffn.ffn_norm
    module.backbone.transformer_blocks.5.node_ffn.ffn_norm
    module.output_heads.energyandforcehead.head.energy_ffn.0.0
    module.output_heads.energyandforcehead.head.energy_ffn.1.0
)
models=(uma allscaip)
seeds=(1 2 3)
args_list=()

for model in "${models[@]}"; do
    if [ ${model} == "uma" ]; then
        features=("${uma_features[@]}")
        precompute_features=true
    else
        features=("${allscaip_features[@]}")
        precompute_features=false
    fi
    all_features=$(IFS=,; echo "[${features[*]}]")
    model_args="\
        data=${model}_features \
        model=${model} \
        method.calibrate.precompute_features=${precompute_features} \
        model_name=${model}.pt"

    for seed in "${seeds[@]}"; do
        for i in "${!features[@]}"; do
            args="${base_args} \
                ${model_args} \
                seed=${seed} \
                data.data_split=$((seed - 1)) \
                model.variance_head.feature_keys=[${features[i]}] \
                model_dir=models/${model}/post_hoc/seed-${seed}/feature-${i}"
            args_list+=("${args}")
        done

        args="${base_args} \
            ${model_args} \
            seed=${seed} \
            data.data_split=$((seed - 1)) \
            model.variance_head.feature_keys=${all_features} \
            model_dir=models/${model}/post_hoc/seed-${seed}/feature-all"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
