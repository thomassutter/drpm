# Variational Clustering

This directory contains the variational clustering experiment from the paper *Differentiable Random Partition Models*.

To run the experiments, please install the environment in `requirements.txt`.

We configure our experiments using **hydra** and recommend checking the file `config.py` to see the different options for all hyperparameters. These can either be passed through `conf/config.yaml` or the command line.
In order to train DRPM-VC or one of the baselines for either MNIST of FMNIST, execute
```
python main.py +experiment=<experiment-name> +dataset=<dataset-name> ++dataset.data_root=<dataset-root>
```
**experiment-name** can be one of *gmm*, *embeded_gmm*, *vade*, or *drpm_clustering*, whereas **dataset-name** can be one of *mnist*, *fashion_mnist* or *stl10*. Make sure to set the correct dataset location through **dataset-root**.

To reproduce the results reported in the main text of the paper, we added the two scripts `run_mnist.sh` and `run_fmnist.sh`. The STL-10 experiments in the appendix can be executed with `run_stl10.sh`.
