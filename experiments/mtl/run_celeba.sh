#!/bin/bash
source ~/.bashrc
enable_modules
conda activate mtl

wandb_entity="PUT YOUR WANDB ENTITY HERE"
project_name="mtl"
dir_experiments="PUT YOUR EXPERIMENTS DIR HERE"
logdir=${dir_experiments}/logs/${project_name}
datadir="PUT YOUR DATA DIR HERE"


device=cuda  # 'cuda' if you are useing a GPU
models=("drpm") # or "baseline"
regularization_weights=(0.015) # or 0.0 for baseline/ULS method
seeds=(0 1 2 3 4)

for model in ${models[@]}; do
for reg_weight in ${regularization_weights[@]}; do
for seed in ${seeds[@]}; do

run_name=${model}_enc_seed_${seed}
wandb_logdir=${logdir}
mkdir -p ${wandb_logdir}

python mtl_main.py \
    model=${model} \
    ++model.device=${device} \
    ++model.seed=${seed} \
    ++model.epochs=50 \
    ++model.drpm_use_encoding_model=True \
    ++model.regularization_weight=${reg_weight} \
    ++dataset.root_dir=${datadir} \
    ++dataset.n_samples=64 \
    ++dataset.n_clusters_model=10 \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_run_name=${run_name} \
    ++log.wandb_group="${model}_enc" \
    ++log.dir_logs=${wandb_logdir} \
    ++log.wandb_project_name=${project_name} \
    ++log.wandb_offline=True
done
done
done
