#!/bin/bash
#SBATCH -o ./logs/run_%A/%a.out
#SBATCH -a 0-4
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
model_names=("dinov2-vit-large-p14" "dino-vit-base-p16" "OpenCLIP" "DreamSim" "vit_b_16")
source_values=("ssl" "ssl" "custom" "custom" "torchvision")
model_parameters_values=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14", "dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"weights":"DEFAULT"}')
module_names=('norm' 'norm' 'visual' 'model.mlp' 'encoder.ln')

model=${model_names[$SLURM_ARRAY_TASK_ID]}
source=${source_values[$SLURM_ARRAY_TASK_ID]}
model_parameters=${model_parameters_values[$SLURM_ARRAY_TASK_ID]}
module_name=${module_names[$SLURM_ARRAY_TASK_ID]}


### Define datasets used and paths to datasets, features and output folder
base_project_path="/home/space/diverse_priors"

# Can define a .txt file or a sequence with several datasets, e.g. ("wds/vtab/pcam" )
dataset="./webdatasets_wo_pcam_svhn.txt"
dataset_root="${base_project_path}/datasets/wds/wds_{dataset_cleaned}"

feature_root="${base_project_path}/features"

output_fn="${base_project_path}/results/single_models/{dataset}_{model}_{task}_{fewshot_k}_seed_{seed}.json"


## Evaluate all datasets with different fewshot settings and seefs on current model (defined by SLURM_ARRAY_TASK_ID).
fewshot_ks=( -1 1 10 100 );
seeds=( {0..9} );

for fewshot_k in "${fewshot_ks[@]}"
do
    for seed in "${seeds[@]}"
    do
        # shellcheck disable=SC2068
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
                            --fewshot_k="$fewshot_k" \
                            --fewshot_lr 0.1 \
                            --fewshot_epochs 20 \
                            --train_split train \
                            --test_split test \
                            --seed="$seed"
    done
done
