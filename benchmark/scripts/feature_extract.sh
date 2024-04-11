#!/bin/bash
#SBATCH -o ./logs/run_%A/%a_out.txt
#SBATCH -e ./logs/run_%A/%a_err.txt
#SBATCH -a 0-179
#SBATCH -J div_prio
#
#SBATCH --partition=gpu-2d
#SBATCH --exclude=head046,head028
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="80gb|40gb"

### Define conda environment 
source /home/lciernik/tools/miniconda3/etc/profile.d/conda.sh
conda activate clip_benchmark

### Model configurations -> see models_config.json
model_names=("dinov2-vit-large-p14" "dino-vit-base-p16" "OpenCLIP" "DreamSim" "vit_b_16");
source_values=("ssl" "ssl" "custom" "custom" "torchvision");
model_parameters_values=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14","dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"extract_cls_token":true,"weights":"DEFAULT"}');
module_names=('norm' 'norm' 'visual' 'model.mlp' 'encoder.ln');

n_models=${#model_names[@]};

### Define datasets used and paths to datasets, features and output folder
base_project_path="/home/space/diverse_priors";

# Can define a .txt file or a sequence with several datasets, e.g. ("wds/vtab/pcam" )
dataset="./webdatasets.txt";
dataset_root="${base_project_path}/datasets/wds/wds_{dataset_cleaned}";

feature_root="${base_project_path}/features";

output_fn="${base_project_path}/results/single_models/{fewshot_k}/{dataset}/{model}/fewshot_lr_{fewshot_lr}/fewshot_epochs_{fewshot_epochs}/seed_{seed}";

# Define different parameter settings. Each combination runs in a separate job of a job array.
# Combinations handled in benchmark code (i.e., cli.py)
fewshot_lrs=( 0.1 0.01 );
fewshot_ks=( -1 10 100 );
fewshot_epochs=( 10 20 30 );
seeds=( {0..9} );

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

for ((i=0; i<"$n_models"; i++))
do
    model="${model_names[i]}"
    source="${source_values[i]}"
    model_parameters="${model_parameters_values[i]}"
    module_name="${module_names[i]}"

    clip_benchmark eval --dataset ${dataset[*]} \
                        --dataset_root=$dataset_root \
                        --feature_root=$feature_root \
                        --output=$output_fn \
                        --task=linear_probe \
                        --model="$model" \
                        --model_source="$source" \
                        --model_parameters="$model_parameters" \
                        --module_name="$module_name" \
                        --batch_size=64 \
                        --fewshot_k ${fewshot_ks[*]} \
                        --fewshot_lr ${fewshot_lrs[*]} \
                        --fewshot_epochs ${fewshot_epochs[*]} \
                        --train_split train \
                        --test_split test \
                        --seed ${seeds[*]}

done


