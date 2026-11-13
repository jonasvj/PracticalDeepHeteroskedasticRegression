#!/bin/bash
#SBATCH --job-name=omol25_post_hoc_ensemble
#SBATCH --array=0-5
#SBATCH --gres=gpu
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=4

base_args="\
    do_training=false \
    do_calibration=false \
    use_var_scale_during_normal_test=true \
    method=post_hoc \
    model.variance_head._target_=src.models.FeatureModelEnsembleVariancePrediction"

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
    else
        features=("${allscaip_features[@]}")
    fi
    all_features=$(IFS=,; echo "[${features[*]}]")

    for seed in "${seeds[@]}"; do
        pretrained_models="["
        for i in "${!features[@]}"; do
            pretrained_models+="models/${model}/post_hoc/seed-${seed}/feature-${i}/${model}.pt,"
        done
        pretrained_models="${pretrained_models%,}]"
        args="${base_args} \
            data=${model}_features \
            model=${model} \
            seed=${seed} \
            data.data_split=$((seed - 1)) \
            model.variance_head.feature_keys=${all_features} \
            model_dir=models/${model}/post_hoc/seed-${seed}/feature-ensemble \
            model_name=${model}.pt \
            pretrained_model=${pretrained_models}"
        args_list+=("${args}")
    done
done

echo Number of array jobs: ${#args_list[@]}
echo Arguments: ${args_list[${SLURM_ARRAY_TASK_ID}]}
python3 experiments/train.py ${args_list[${SLURM_ARRAY_TASK_ID}]}
