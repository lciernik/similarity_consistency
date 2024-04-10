#!/bin/bash

# define base path experiment
current_datetime=$(date +"%Y_%m_%d_%H_%M");
base_path="/home/lciernik/projects/divers-priors/results_local/tests";
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

i=1;
j=0;
sbatch feature_extract.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                          "${model_names[i]}" "${source_values[i]}" "${model_parameters_values[i]}" "${module_names[i]}" "" \
                          "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed"

## gLocal application and normalization 
i=1;
j=1;
sbatch feature_extract.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                          "${model_names[i]}" "${source_values[i]}" "${model_parameters_values[i]}" "${module_names[i]}" "gLocal" \
                          "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed"

i=2;
j=1;
sbatch feature_extract.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                          "${model_names[i]}" "${source_values[i]}" "${model_parameters_values[i]}" "${module_names[i]}" "gLocal" \
                          "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed"


# ### combined features 

output_fn="${output_root}/combined_models/{fewshot_k}/{feature_combiner}/{dataset}/{model}_{feature_alignment}/fewshot_lr_{fewshot_lr}/fewshot_epochs_{fewshot_epochs}/seed_{seed}"

models_subset=("${model_names[@]:0:2}")
sources_subset=("${source_values[@]:0:2}")
params_subset=("${model_parameters_values[@]:0:2}")
modules_subset=("${module_names[@]:0:2}")

j=0;
## not normalized and concat (each representation is normalized)
sbatch combined_features.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                             "${models_subset[*]}" "${sources_subset[*]}" "${params_subset[*]}" "${modules_subset[*]}" "" \
                             "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed" "concat" "False"

## only normalized and concat (each representation is normalized)
sbatch combined_features.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                             "${models_subset[*]}" "${sources_subset[*]}" "${params_subset[*]}" "${modules_subset[*]}" "" \
                             "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed" "concat" "True"


models_subset=("${model_names[@]:1:2}")
sources_subset=("${source_values[@]:1:2}")
params_subset=("${model_parameters_values[@]:1:2}")
modules_subset=("${module_names[@]:1:2}")


j=1;
## gLocal and concat
sbatch combined_features.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                             "${models_subset[*]}" "${sources_subset[*]}" "${params_subset[*]}" "${modules_subset[*]}" "gLocal" \
                             "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed" "concat" "False"

## gLocal and concat_pca 
sbatch combined_features.sh "$dataset" "$dataset_root" "$feature_root" "$output_fn" \
                             "${models_subset[*]}" "${sources_subset[*]}" "${params_subset[*]}" "${modules_subset[*]}" "gLocal" \
                             "$fewshot_lr" "${fewshot_ks[j]}" "$fewshot_epoch" "$seed" "concat_pca" "False"








