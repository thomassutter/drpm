from dataclasses import dataclass, field
from typing import List, Dict, Optional

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    root_dir: str = MISSING
    # tasks
    tasks: List[str] = MISSING
    # model config
    n_clusters_model: int = MISSING
    n_samples: int = MISSING

    # Number of workers for data loaders
    num_workers: int = 8

    # Eval on all noise ratios
    eval_noise_ratios: bool = False


@dataclass
class MNISTDatasetConfig(DatasetConfig):
    name: str = "mnist"
    root_dir: str = "PUT DATA DIR MULTIMNIST HERE"
    # tasks
    tasks: List[str] = field(default_factory=lambda: ["L", "R"])
    # model config
    n_clusters_model: int = 2
    n_samples: int = 50

    # Amount of `pepper` noise added to the right digit
    right_noise_p: float = 0

    # Minimum amount of `pepper` noise added to both digits
    both_noise_min: Optional[float] = None


@dataclass
class CelebADatasetConfig(DatasetConfig):
    name: str = "celeba"
    # root_dir: str = "/tmp/data/CelebA"
    root_dir: str = "PUT DATA DIR CELEBA HERE"

    # model config
    n_clusters_model: int = 40
    n_samples: int = 64
    # tasks
    tasks: List[str] = field(default_factory=lambda: [str(t) for t in range(0, 40)])

    # dataset specific
    img_rows: int = 64
    img_cols: int = 64


@dataclass
class ModelConfig:
    # Model name (drpm, baseline)
    name: str = MISSING
    # train config
    seed: int = 0
    device: str = "cuda"
    # general
    batch_size: int = 128
    epochs: int = 50
    lr: float = 0.0001
    # gumbel
    resample: bool = False
    hard: bool = True
    # annealing
    annealing_type: str = "exp"
    final_temp: float = 0.5
    start_temp: float = 1.0
    num_steps_annealing: int = 100000

    # When to split the tasks. "dense" or "conv"
    tasks_split_layer: str = "dense"

    # Regularization weight encourages sparsity of latent representations
    regularization_weight: float = 0.015

    checkpoint_metric: str = "accuracy/mean_tasks_acc"


@dataclass
class MTLDRPMModelConfig(ModelConfig):
    name: str = "drpm"
    reverse_partitions_order: bool = False
    learn_const_omegas: bool = False
    learn_const_scores: bool = False
    const_omegas_initializer: str = "normal"
    const_scores_initializer: str = "normal"
    regularization_partition_index: Optional[int] = None
    drpm_use_encoding_model: bool = False


@dataclass
class MTLBaselineModelConfig(ModelConfig):
    name: str = "baseline"


@dataclass
class MTLManualPartitionConfig(ModelConfig):
    name: str = "manual_partition_baseline"
    left_partition_p: float = 0.5


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "drpm"
    wandb_group: str = "mtl"
    wandb_run_name: str = "mtl"
    wandb_project_name: str = "mtl"
    wandb_log_freq: int = 50
    wandb_offline: bool = False

    # logs
    dir_logs: str = "PUT LOG DIR HERE"


@dataclass
class MyMTLConfig:
    # Dataset Config
    dataset: DatasetConfig = MISSING
    # model
    model: ModelConfig = MISSING
    # logger
    log: LogConfig = MISSING
