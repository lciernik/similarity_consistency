#!/bin/bash
#SBATCH -o /home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/logs/run_%A/%a.out
#SBATCH -a 0-5 
#SBATCH -J div_prio
#
#SBATCH --partition=gpu-2d
#SBATCH --exclude=head046,head028
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100000M

source /home/lciernik/tools/miniconda3/etc/profile.d/conda.sh
conda activate clip_benchmark

# Model configurations -> see models_config.json

model_names=("dinov2-vit-large-p14" "dino-vit-base-p16" "OpenCLIP" "OpenCLIP" "DreamSim" "vit_b_16")

#pretrained_values=("yes" "imagenet" "laion400m_e32" "laion400m_e32" "yes" "imagenet")

source_values=("ssl" "ssl" "custom" "custom" "custom" "torchvision")

model_parameters_values=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14", "dataset":"laion400m_e32"}' '{"variant":"ViT-L-14-quickgelu", "dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"weights":"DEFAULT"}')

module_names=('norm' 'norm' 'visual' 'visual' 'model.mlp' 'encoder.ln')

model=${model_names[$SLURM_ARRAY_TASK_ID]}
#pretrained=${pretrained_values[$SLURM_ARRAY_TASK_ID]}
source=${source_values[$SLURM_ARRAY_TASK_ID]}
model_parameters=${model_parameters_values[$SLURM_ARRAY_TASK_ID]}
module_name=${module_names[$SLURM_ARRAY_TASK_ID]}

base_project_path="/home/space/diverse_priors"

dataset="/home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/webdatasets.txt"
dataset_root="${base_project_path}/datasets/wds/wds_{dataset_cleaned}"
# dataset_root="https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \

feature_root="${base_project_path}/features"

output_fn="${base_project_path}/results/single_models/{dataset}_{model}_{task}_${SLURM_ARRAY_JOB_ID}.json"

# shellcheck disable=SC2068
clip_benchmark eval --dataset=$dataset \
                    --dataset_root=$dataset_root \
                    --feature_root=$feature_root \
                    --output=$output_fn \
                    --task=linear_probe \
                    --model="$model" \
                    --model_source="$source" \
                    --model_parameters="$model_parameters" \
                    --module_name="$module_name" \
                    --batch_size=64 \
                    --fewshot_lr 0.1 \
                    --fewshot_epochs 20 \
                    --train_split train \
                    --test_split test


#                    --pretrained="$pretrained" \
