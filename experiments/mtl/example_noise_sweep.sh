#!/bin/bash

wandb_entity=YOUR_WANDB_ENTITY
datedir=$(date +%Y-%m-%d-%H.%M.%S)
project_name=mtl_noisy_multimnist
logdir=$(pwd)/logs/${project_name}/${datedir}


device=cpu  # 'cuda' if you are useing a GPU
right_noise_p_values=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
models=(drpm baseline)
seeds=(0 1 2 4 5)

for right_noise_p in ${right_noise_p_values[@]}; do
for model in ${models[@]}; do
for seed in ${seeds[@]}; do

run_name=submit_${model}_enc_rnoise_${right_noise_p}_seed_${seed}
current_logdir=${logdir}/${run_name}
wandb_logdir=${current_logdir}/wandb
mkdir -p ${wandb_logdir}

python mtl_main.py \
    ++model.device=${device} \
    ++model.seed=${seed} \
    ++model.epochs=200 \
    ++model.drpm_use_encoding_model=True \
    ++dataset.right_noise_p=${right_noise_p} \
    ++dataset.n_samples=50 \
    ++dataset.n_clusters_model=2 \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_run_name=${run_name} \
    ++log.wandb_group=${model}_enc \
    ++log.dir_logs="${wandb_logdir}" \
    ++log.wandb_project_name=${project_name} \


done
done
done

