# DRPM Multitask Learning Experiment

This folder contains the code of the multitask learning experiments.
The script `example_noise_sweep.sh` shows how the experiment can be run.
To enable evaluation on all noise ratios of the noisyMultiMNIST dataset, add the parameter `++model.eval_noise_ratios=true`.

## Important Parameters

| Parameter                                     | Description                                      | 
|-----------------------------------------------|--------------------------------------------------|
| `++dataset.eval_noise_ratios=true`            | Enable evaluation on noise ratios 0, 0.1, ..., 1 |
| `++dataset.right_noise_p=<Noise Ratio>`       | Set the training noise ratio |
| `++dataset.n_clusters_model=<Num Partitions>` | Number of partitions of the latent representation |
| `++dataset.n_samples`                         | Size of the latent representation |
| `++model.drpm_use_encoding_model=true`        | Infer the partitioning based on the input instead of the latent representation |




