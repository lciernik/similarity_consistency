#!/bin/bash

# define base path experiment
current_datetime=$(date +"%Y_%m_%d_%H_%M");
base_path="/home/jonasd/code/diverse_priors/benchmark/results_local/tests";
base_path="${base_path}/${current_datetime}"
mkdir -p "$base_path"
# base_path="/home/lciernik/projects/divers-priors/results_local/tests/2024_04_09_10_36"

# define feature and output path
feature_root="${base_path}/features";
mkdir -p "$feature_root"

output_root="${base_path}/results";
mkdir -p "$output_root"

# define datasets and features path
dataset="./webdatasets_test.txt";
dataset_root="/home/space/diverse_priors/datasets/wds/wds_{dataset_cleaned}";

## all models
model_names=("dinov2-vit-large-p14" "dino-vit-base-p16" "OpenCLIP" "DreamSim" "vit_b_16");
source_values=("ssl" "ssl" "custom" "custom" "torchvision");
model_parameters_values=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14","dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"extract_cls_token":true,"weights":"DEFAULT"}');
module_names=('norm' 'norm' 'visual' 'model.mlp' 'encoder.ln');


fewshot_lr=0.1;
fewshot_epoch=10;
seed=7;
fewshot_ks=(-1 10);


#### Define different tests (different models, lr, fewshot ks, fewshot epochs, seeds, human alignement, feature combiners )
### single features

output_fn="${output_root}/single_models/{fewshot_k}/{dataset}/{model}_{feature_alignment}/fewshot_lr_{fewshot_lr}/fewshot_epochs_{fewshot_epochs}/seed_{seed}";

## only normalized
i=0;
j=0;
sbatch feature_extract.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                          "${model_names[i]}" "${source_values[i]}" "${model_parameters_values[i]}" "${module_names[i]}" "" \
                          "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed"








