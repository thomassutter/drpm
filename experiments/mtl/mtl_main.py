import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

import torch.multiprocessing

# import clustering_config as cfg
from omegaconf import MISSING, OmegaConf
from drpm_mtl_module import DRPMmtl
from mtl_baseline_model import MTLBaseline
from mtl_manual_partition_model import MTLManualPartitionBaseline
from MyMTLConfig import MyMTLConfig
from MyMTLConfig import MNISTDatasetConfig
from MyMTLConfig import CelebADatasetConfig
from MyMTLConfig import ModelConfig
from MyMTLConfig import LogConfig
from MyMTLConfig import MTLBaselineModelConfig
from MyMTLConfig import MTLDRPMModelConfig
from MyMTLConfig import MTLManualPartitionConfig
from utils import utils as utils

OmegaConf.register_new_resolver("balance_task_weights", lambda left: 1 - float(left))

torch.multiprocessing.set_sharing_strategy("file_system")
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(group="dataset", name="mnist", node=MNISTDatasetConfig)
cs.store(group="dataset", name="celeba", node=CelebADatasetConfig)
cs.store(group="model", name="drpm", node=MTLDRPMModelConfig)
cs.store(group="model", name="baseline", node=MTLBaselineModelConfig)
cs.store(group="model", name="manual_partition_baseline", node=MTLManualPartitionConfig)
cs.store(group="log", name="log", node=LogConfig)
cs.store(name="base_config", node=MyMTLConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyMTLConfig):
    print(cfg)
    pl.utilities.seed.seed_everything(cfg.model.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, val_dst = utils.get_dataset(cfg)

    # get loss function
    loss_fn = utils.get_loss(cfg)

    networks = utils.get_networks(cfg)

    if cfg.dataset.name == "mnist":
        training_noise = (
            cfg.dataset.both_noise_min
            if cfg.dataset.both_noise_min is not None
            else cfg.dataset.right_noise_p
        )
    else:
        training_noise = 0

    # init model
    model = None
    if cfg.model.name == "drpm":
        model = DRPMmtl(
            cfg.dataset.tasks,
            loss_fn,
            networks,
            cfg.dataset.n_samples,
            cfg.dataset.n_clusters_model,
            lr=cfg.model.lr,
            seed=cfg.model.seed,
            task_split_layer=cfg.model.tasks_split_layer,
            learn_const_omegas=cfg.model.learn_const_omegas,
            learn_const_scores=cfg.model.learn_const_scores,
            const_omegas_initializer=cfg.model.const_omegas_initializer,
            const_scores_initializer=cfg.model.const_scores_initializer,
            reverse_partitions_order=cfg.model.reverse_partitions_order,
            resample=cfg.model.resample,
            hard=cfg.model.hard,
            final_temp=cfg.model.final_temp,
            start_temp=cfg.model.start_temp,
            num_steps_annealing=cfg.model.num_steps_annealing,
            annealing_type=cfg.model.annealing_type,
            regularization_partition_index=cfg.model.regularization_partition_index,
            regularization_weight=cfg.model.regularization_weight,
            drpm_use_encoding_model=cfg.model.drpm_use_encoding_model,
            eval_noise_ratios=cfg.dataset.eval_noise_ratios,
            training_noise=training_noise,
            device=cfg.model.device,
        )
    elif cfg.model.name == "baseline":
        model = MTLBaseline(
            cfg.dataset.tasks,
            loss_fn,
            networks,
            n_partition_samples=cfg.dataset.n_samples,
            lr=cfg.model.lr,
            eval_noise_ratios=cfg.dataset.eval_noise_ratios,
            training_noise=training_noise,
            seed=cfg.model.seed,
            regularization_weight=cfg.model.regularization_weight,
            device=cfg.model.device,
        )
    elif cfg.model.name == "manual_partition_baseline":
        model = MTLManualPartitionBaseline(
            cfg.dataset.tasks,
            loss_fn,
            networks,
            cfg.dataset.n_samples,
            cfg.model.left_partition_p,
            lr=cfg.model.lr,
            eval_noise_ratios=cfg.dataset.eval_noise_ratios,
            training_noise=training_noise,
            seed=cfg.model.seed,
            regularization_weight=cfg.model.regularization_weight,
            device=cfg.model.device,
        )

    assert model is not None

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log.dir_logs,
        monitor=cfg.model.checkpoint_metric,
        mode="max",
        save_last=True,
    )
    str_project = cfg.log.wandb_project_name + "_" + cfg.dataset.name
    wandb_logger = WandbLogger(
        # name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=str_project,
        group=cfg.model.name,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        devices=1,
        accelerator="gpu" if cfg.model.device == "cuda" else cfg.model.device,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        deterministic=True,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    for ind in range(model.final_accuracies.shape[0]):
        model.logger.log_metrics(
            {f"final_scores/final_acc_task_%d" % ind: model.final_accuracies[ind]}
        )
    model.logger.log_metrics({f"final_scores/final_mean_acc": model.final_mean_acc})


if __name__ == "__main__":
    run_experiment()
