#!/bin/bash
#SBATCH -o ./logs/run_%A/%a.out
#SBATCH -a 0-159
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
source /home/lciernik/tools/miniconda3/bin/activate
conda activate clip_benchmark

### Model configurations -> see models_config.json
model=("dinov2-vit-large-p14" "dino-vit-base-p16" "OpenCLIP" "DreamSim" "vit_b_16")
source=("ssl" "ssl" "custom" "custom" "torchvision")
model_parameters=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14","dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"weights":"DEFAULT"}')
module_name=('norm' 'norm' 'visual' 'model.mlp' 'encoder.ln')

### Define paths 
base_project_path="/home/space/diverse_priors"

dataset="./webdatasets.txt"
dataset_root="${base_project_path}/datasets/wds/wds_{dataset_cleaned}"

feature_root="${base_project_path}/features"

output_fn="${base_project_path}/results/combined_models/{fewshot_k}/{feature_combiner}/{dataset}_{model}_{task}_seed_{seed}.json"


### Define different parameter settings. Each combination run in a separate job of a job array
fewshot_ks=( -1 1 10 100 );
seeds=( {0..9} );
combiners=("concat" "concat_pca");
fewshot_lrs=(0.01 0.001)

# Calculate the index for each array
fewshot_lr_index=$(( ($SLURM_ARRAY_TASK_ID/(${#combiners[@]} * ${#seeds[@]} * ${#fewshot_ks[@]})) %  ${#fewshot_lrs[@]})); 
fewshot_k_index=$(( ($SLURM_ARRAY_TASK_ID/(${#combiners[@]} * ${#seeds[@]} )) % ${#fewshot_ks[@]} )); 
seed_index=$(( ($SLURM_ARRAY_TASK_ID/${#combiners[@]}) % ${#seeds[@]})); 
combiner_index=$(( $SLURM_ARRAY_TASK_ID % ${#combiners[@]}));


# Get the elements for the current combination
fewshot_k=${fewshot_ks[$fewshot_k_index]}
seed=${seeds[$seed_index]}
feature_combiner=${combiners[$combiner_index]}
fewshot_lr=${fewshot_lrs[$fewshot_lr_index]}


clip_benchmark eval --dataset ${dataset} \
                    --dataset_root=$dataset_root \
                    --feature_root=$feature_root \
                    --output=$output_fn \
                    --task=linear_probe \
                    --model ${model[*]} \
                    --model_source ${source[*]} \
                    --model_parameters ${model_parameters[*]} \
                    --module_name ${module_name[*]} \
                    --batch_size=64 \
                    --fewshot_k="$fewshot_k" \
                    --fewshot_lr="$fewshot_lr" \
                    --fewshot_epochs 20 \
                    --train_split train \
                    --test_split test \
                    --seed="$seed" \
                    --eval_combined \
                    --feature_combiner "$feature_combiner"
