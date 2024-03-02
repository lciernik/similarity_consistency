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


# Function to subset array based on indices
subset_array() {
    local array_name=("${!1}")
    local subset_indices=("${!2}")

    sub=''
    for index in "${subset_indices[@]}"; do
        sub="${sub} ${array_name[$index]}"
    done

    echo "${sub}"
}

source /home/lciernik/tools/miniconda3/etc/profile.d/conda.sh
conda activate clip_benchmark

# Model configurations -> see models_config.json
model_names=("dinov2-vit-large-p14" "dino-vit-base-p16" "OpenCLIP" "DreamSim" "vit_b_16")
source_values=("ssl" "ssl" "custom" "custom" "torchvision")
model_parameters_values=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14", "dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"weights":"DEFAULT"}')
module_names=('norm' 'norm' 'visual' 'model.mlp' 'encoder.ln')


# Get the indices from the idx_combinations.txt file
indices=($(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" idx_combinations.txt))

# Subset the arrays based on the indices
model=$(subset_array model_names[@] indices[@])
source=$(subset_array source_values[@] indices[@])
model_parameters=$(subset_array model_parameters_values[@] indices[@])
module_name=$(subset_array module_names[@] indices[@])

base_project_path="/home/space/diverse_priors"

dataset="/home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/webdatasets.txt"
dataset_root="${base_project_path}/datasets/wds/wds_{dataset_cleaned}"

feature_root="${base_project_path}/features"

output_fn="${base_project_path}/results/combined_models/{dataset}_{model}_{task}_${SLURM_ARRAY_JOB_ID}.json"

# shellcheck disable=SC2068
clip_benchmark eval --dataset=$dataset \
                    --dataset_root=$dataset_root \
                    --feature_root=$feature_root \
                    --output=$output_fn \
                    --task=linear_probe \
                    --model="${model[*]}" \
                    --model_source="${source[*]}" \
                    --model_parameters="${model_parameters[*]}" \
                    --module_name="${module_name[*]}" \
                    --batch_size=64 \
                    --fewshot_lr 0.1 \
                    --fewshot_epochs 20 \
                    --train_split train \
                    --test_split test


#                    --pretrained="$pretrained" \
