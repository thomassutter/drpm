from dataclasses import dataclass
from omegaconf import MISSING

    
@dataclass
class LogConfig:
    name: str ="log"
    # logs
    dir_logs: str = "/tmp/wandb_logs"
    # validation logging frequency
    val_freq: int = 64
    # wandb
    wandb_project_name: str = "clustering"
    wandb_run_name: str = ''
    wandb_log_freq: int = 50
    # Put checkpoints on wandb
    wandb_checkpoints: bool = False
    # Log gradients and parameters during training
    wandb_watch: bool = False
    # Whether to use pca instead of umap for visualization
    use_pca: bool = False

@dataclass
class ExperimentConfig:
    name: str  = MISSING
    # train config
    # general
    batch_size: int = 256
    epochs: int = 1024
    lr: float = 1e-4
    # gumbel
    resample: bool = True
    hard: bool = True
    # annealing
    final_temp: float = 0.5
    min_temp_step: int = 100000
    init_temp: float = 1.
    tau_schedule: str = 'exp'
    # model config
    n_clusters_model: int = 10
    # beta, gamma and delta weightings of the DRPM-VC loss
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 0.01
    # Whether to retrain pretrained model even if checkpoints exist already
    retrain_pretrain: bool = False
    # intermediate and latent dimension for autoencoders
    intermediate_dim: int = 2000
    l_dim: int = 10
    # Set the metric to look for when checkpointing
    checkpoint_metric: str = 'validation_score/rand_score_cluster_prob'
    # prior initialization strategy, can be either 'pretrain' (reconstruction) or 'pretrain_tcl'
    prior_init: str = 'pretrain_tcl'
    reinit_priors: bool = False

@dataclass
class GMMConfig(ExperimentConfig):
    name: str = 'gmm'
    prior_init: str = 'pretrain'
    
@dataclass
class EmbededGMMConfig(ExperimentConfig):
    name: str = 'embeded_gmm'
    
@dataclass
class DRPMClusteringConfig(ExperimentConfig):
    name: str = 'drpm_clustering'
    
@dataclass
class VaDEConfig(ExperimentConfig):
    # Set to true if you want to optimize like in the original VADE paper
    vanilla_vade: bool = False
    name: str = 'vade'
    
@dataclass
class DataConfig:
    name: str = MISSING
    data_root: str = "datasets"
    
@dataclass
class MNISTDataConfig(DataConfig):
    """
    Configure mnist
    """
    name: str ="mnist"
    inp_dim: int = 784
    
@dataclass
class FashionMNISTDataConfig(MNISTDataConfig):
    """
    Configure fashion mnist
    """
    name: str ="fashion_mnist"
    inp_dim: int = 784
    
@dataclass
class STL10DataConfig(DataConfig):
    """
    Configure stl10 Dataset
    """
    name: str ="stl10"
    inp_dim: int = 2048
    l_dim: int = 20
    
@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    # Configure logging
    logging: LogConfig = LogConfig
    # Configure which experiment to take
    experiment: ExperimentConfig = MISSING
    # Configure dataset
    dataset: DataConfig = MISSING

    

    

    
