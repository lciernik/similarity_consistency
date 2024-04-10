#!/bin/bash
#SBATCH -o ./logs/run_%A/%a_out.txt
#SBATCH -e ./logs/run_%A/%a_err.txt
#SBATCH -a 0
#SBATCH -J comb_mod_div
#
#SBATCH --partition=gpu-2h
#SBATCH --exclude=head046,head028
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="80gb|40gb"

## Get arguments
dataset=$1;
dataset_root=$2;
feature_root=$3;
output_fn=$4;

model=$5;
source=$6;
model_parameters=$7;
module_name=$8;
feature_alignment=$9;

fewshot_lr=${10}; 
fewshot_k=${11};
fewshot_epoch=${12}; 
seed=${13};

feat_combiner=${14};
normalize=${15};

### waiting for single model jobs to be done
job_name_to_wait_for="sing_mod_div"
USER="lciernik" # TODO change accordingly 
while true; do
    # Check the status of jobs with the specified job name
    pending_jobs=$(squeue -h -u "$USER" -n "$job_name_to_wait_for" -t PD | wc -l)
    running_jobs=$(squeue -h -u "$USER" -n "$job_name_to_wait_for" -t R | wc -l)

    # If there are no pending or running jobs, exit the loop
    if [ "$pending_jobs" -eq 0 ] && [ "$running_jobs" -eq 0 ]; then
        break
    fi

    # Sleep for a while before checking again
    echo "Sleeping for 60s to wait until single model runs are done ..."
    sleep 60  # You can adjust the sleep time as needed
    
done


### Define conda environment: TODO change accordingly 
source /home/lciernik/tools/miniconda3/etc/profile.d/conda.sh
conda activate clip_benchmark

## Define parameters
fewshot_lrs=( "$fewshot_lr" );
fewshot_ks=( "$fewshot_k" );
fewshot_epochs=( "$fewshot_epoch"  );
seeds=( "$seed" );

## Export global parameter
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform


clip_benchmark eval --dataset ${dataset} \
                    --dataset_root=$dataset_root \
                    --feature_root=$feature_root \
                    --output=$output_fn \
                    --task=linear_probe \
                    --model ${model[*]} \
                    --model_source ${source[*]} \
                    --model_parameters ${model_parameters[*]} \
                    --module_name ${module_name[*]} \
                    --feature_alignment="$feature_alignment" \
                    --batch_size=64 \
                    --fewshot_k "${fewshot_ks[*]}" \
                    --fewshot_lr "${fewshot_lrs[*]}" \
                    --fewshot_epochs "${fewshot_epochs[*]}" \
                    --normalize "$normalize" \
                    --train_split train \
                    --test_split test \
                    --seed "${seeds[*]}" \
                    --eval_combined \
                    --feature_combiner "$feat_combiner"
