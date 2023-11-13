#!/bin/bash
LOG_DIR="./logs"
mkdir $LOG_DIR

data_dir="/datasets" # Set this to parent directory of the dataset

exp_name=stl10_experiments

dir_logs="${LOG_DIR}/${exp_name}"
mkdir $dir_logs
mkdir $dir_logs/outputs/
mkdir $dir_logs/wandb/

experiment=drpm_clustering #gmm embded_gmm vade

n_cluster=10

num_splits=5

for i in $(seq 1 $num_splits)
do
    seed=$(( 42*i ))
    name="${experiment}-vc_stl10_seed_${i}"
    python main.py \
    +experiment=$experiment \
    +dataset=stl10 \
    ++dataset.data_root=${data_dir} \
    ++experiment.epochs=256 \
    ++experiment.batch_size=32 \
    ++experiment.prior_init="pretrain" \
    ++seed=${seed} \
    ++logging.wandb_project_name=${exp_name} \
    ++logging.wandb_run_name=${name} \
    ++logging.dir_logs=${dir_logs} \
    ++logging.val_freq=8 \
    ++experiment.reinit_priors=True \
    ++experiment.n_clusters_model=${n_cluster} \
    ++experiment.intermediate_dim=4000
done