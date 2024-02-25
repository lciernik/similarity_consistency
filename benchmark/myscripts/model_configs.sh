#!/bin/bash

# Read JSON from file
json_file="models_config_small.json"
json=$(<"$json_file")

# Extract model names
model_names=$(echo "$json" | grep -o '"model_name": "[^"]*' | cut -d'"' -f4)
model_names=$(echo "$model_names" | tr '\n' ' ')
model_names=($model_names)

pretrained_values=$(echo "$json" | grep -o '"pretrained": "[^"]*' | cut -d'"' -f4)
pretrained_values=$(echo "$pretrained_values" | tr '\n' ' ')
pretrained_values=($pretrained_values)

source_values=$(echo "$json" | grep -o '"source": "[^"]*' | cut -d'"' -f4)
source_values=$(echo "$source_values" | tr '\n' ' ')
source_values=($source_values)

model_parameters_values=$(echo "$json" | grep -o '"model_parameters": "[^"]*' | cut -d'"' -f4)
model_parameters_values=$(echo "$model_parameters_values" | tr '\n' ' ')
model_parameters_values=($model_parameters_values)

module_names=$(echo "$json" | grep -o '"module_name": "[^"]*' | cut -d'"' -f4)
module_names=$(echo "$module_names" | tr '\n' ' ')
module_names=($module_names)

echo "Model names: ${model_names[*]}"
echo "Pretrained values: ${pretrained_values[*]}"
echo "Source values: ${source_values[*]}"
echo "Model parameters values: ${model_parameters_values[*]}"
echo "Module names: ${module_names[*]}"

#cd /Users/lciernik/Documents/projects/divers_prios/diverse_priors/benchmark
#
clip_benchmark eval --dataset=cifar10 \
                    --task=linear_probe \
                    --pretrained ${pretrained_values[*]} \
                    --model ${model_names[*]} \
                    --output=result.json \
                    --batch_size=64 \
                    --fewshot_lr 0.1 \
                    --fewshot_epochs 20 \
                    --batch_size 512 \
                    --train_split train \
                    --test_split test \
                    --model_source ${source_values[*]} \
                    --model_parameters ${model_parameters_values[*]} \
                    --module_name ${module_names[*]}