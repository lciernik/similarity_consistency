#!/bin/bash
#SBATCH -o /home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/logs/run_%A/%a.out
#SBATCH -a 0-19
#SBATCH -J div_prio
#
#SBATCH --partition=gpu-5h
#SBATCH --exclude=head046
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100000M


dataset_root="/home/space/diverse_priors/datasets/{dataset}"
feature_root="/home/space/diverse_priors/features"

output_fn="/home/space/diverse_priors/results/single_models/{dataset}_{model}_{pretrained}_{model_source}_{model_parameters}_{module_name}_${SLURM_ARRAY_TASK_ID}.json"

# Datasets
datasets=("cifar10" "caltech101" "cifar10" "cifar100" "clevr_count_all" "clevr_closest_object_distance" "diabetic_retinopathy" "dmlab" "dsprites_label_orientation" "dsprites_label_x_position" "dtd" "eurosat" "kitti_closest_vehicle_distance" "flowers" "pets" "pcam" "resisc45" "smallnorb_label_azimuth" "smallnorb_label_elevation" "svhn")

# Model configurations
#pretrained_values=("yes" "yes")
#model_names=("dinov2-vit-large-p14" "DreamSim")
#source_values=("ssl" "custom")
#model_parameters_values=('{"extract_cls_token":true}' '{"variant":"open_clip_vitb32"}')
#module_names=('norm' 'model.mlp')

pretrained_values=("yes")
model_names=("dinov2-vit-large-p14")
source_values=("ssl" )
model_parameters_values=('{"extract_cls_token":true}')
module_names=('norm')

dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#datasets[@]}))
model_idx=$(((SLURM_ARRAY_TASK_ID / ${#datasets[@]}) % ${#model_names[@]}))

dataset=${datasets[$dataset_idx]}
pretrained=${pretrained_values[$model_idx]}
model=${model_names[$model_idx]}
source=${source_values[$model_idx]}
model_parameters=${model_parameters_values[$model_idx]}
module_name=${module_names[$model_idx]}

random_number=$((RANDOM % 6))
echo "waiting $random_number minutes"
seconds=$((random_number * 60 + RANDOM % 30))
sleep $seconds

source /home/lciernik/tools/miniconda3/etc/profile.d/conda.sh
conda activate clip_benchmark

# Predefined arrays with values from JSON
#pretrained_values=("yes" "imagenet" "laion400m_e32" "laion400m_e32" "yes" "imagenet")
#model_names=("dinov2-vit-large-p14" "dino-vit-base-p16" "clip_vitL" "clip_vitL_quickgelu" "dreamsim_open_clip" "vit_b_16")
#source_values=("ssl" "ssl" "custom" "custom" "custom" "torchvision")
#model_parameters_values=('{"extract_cls_token":true}' '{"extract_cls_token":true}' '{"variant":"ViT-L-14", "dataset":"laion400m_e32"}' '{"variant":"ViT-L-14-quickgelu", "dataset":"laion400m_e32"}' '{"variant":"open_clip_vitb32"}' '{"weights":"DEFAULT"}')
#module_names=('norm' 'norm' 'visual' 'visual' 'model.mlp' 'encoder.ln')


# shellcheck disable=SC2068
clip_benchmark eval --dataset_root=$dataset_root \
                    --feature_root=$feature_root \
                    --output=$output_fn \
                    --dataset="$dataset" \
                    --task=linear_probe \
                    --pretrained="$pretrained" \
                    --model="$model" \
                    --batch_size=64 \
                    --fewshot_lr 0.1 \
                    --fewshot_epochs 20 \
                    --train_split train \
                    --test_split test \
                    --model_source="$source" \
                    --model_parameters="$model_parameters" \
                    --module_name="$module_name"



#                    --fewshot_k 5 \
