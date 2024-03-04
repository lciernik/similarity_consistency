#!/bin/bash
#SBATCH -o /home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/logs/run_%A/%a.out
#SBATCH -a 0-39
#SBATCH -J div_prio
#
#SBATCH --partition=gpu-2d
#SBATCH --exclude=head046,head028
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="80gb|40gb"


source /home/lciernik/tools/miniconda3/etc/profile.d/conda.sh
conda activate clip_benchmark

# Model configurations -> see models_config.json
# model_names=("dinov2-vit-large-p14" "dino-vit-base-p16" "OpenCLIP" "DreamSim" "vit_b_16")
# source_values=("ssl" "ssl" "custom" "custom" "torchvision")
# model_parameters_values=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14", "dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"weights":"DEFAULT"}')
# module_names=('norm' 'norm' 'visual' 'model.mlp' 'encoder.ln')

model=("dinov2-vit-large-p14" "OpenCLIP" "DreamSim")
source=("ssl" "custom" "custom" )
model_parameters=('{"extract_cls_token":true}' '{"variant":"ViT-L-14", "dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' )
module_name=('norm' 'visual' 'model.mlp' )

base_project_path="/home/space/diverse_priors"

dataset="/home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/webdatasets_small.txt"
dataset_root="${base_project_path}/datasets/wds/wds_{dataset_cleaned}"

feature_root="${base_project_path}/features"

output_fn="${base_project_path}/results/single_models/{dataset}_{model}_{task}_{fewshot_k}_seed_{seed}.json"

fewshot_ks=( -1 1 10 100 );
seeds=( {0..9} );

# Calculate the index for each array
fewshot_k_index=$(( $SLURM_ARRAY_TASK_ID / ${#seeds[@]} ))
seed_index=$(( $SLURM_ARRAY_TASK_ID % ${#seeds[@]} ))

# Get the elements for the current combination
fewshot_k=${fewshot_ks[$fewshot_k_index]}
seed=${seeds[$seed_index]}

# shellcheck disable=SC2068
clip_benchmark eval --dataset ${dataset} \
                    --dataset_root=$dataset_root \
                    --feature_root=$feature_root \
                    --output=$output_fn \
                    --task=linear_probe \
                    --model ${model[*]} \
                    --model_source ${model[*]} \
                    --model_parameters ${model[*]} \
                    --module_name${model[*]} \
                    --batch_size=64 \
                    --fewshot_k="$fewshot_k" \
                    --fewshot_lr 0.1 \
                    --fewshot_epochs 20 \
                    --train_split train \
                    --test_split test \
                    --seed="$seed"
