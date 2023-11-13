from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import umap
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.drpm_variational_cluster_module import VariationalDRPMClustering
from models.vade import VaDE
from config import *
from data.data import *
from tqdm.auto import tqdm

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(8)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'. 
cs.store(group="experiment", name=EmbededGMMConfig.name, node=EmbededGMMConfig)
cs.store(group="experiment", name=GMMConfig.name, node=GMMConfig)
cs.store(group="experiment", name=DRPMClusteringConfig.name, node=DRPMClusteringConfig)
cs.store(group="experiment", name=VaDEConfig.name, node=VaDEConfig)
cs.store(group="dataset", name=MNISTDataConfig.name, node=MNISTDataConfig)
cs.store(group="dataset", name=FashionMNISTDataConfig.name, node=FashionMNISTDataConfig)
cs.store(group="dataset", name=STL10DataConfig.name, node=STL10DataConfig)
cs.store(group="logging", name=LogConfig.name, node=LogConfig)
cs.store(name="base_config", node=Config)

def cluster_acc(Y, Y_pred):
    """
    From https://github.com/GuHongyang/VaDE-pytorch
    """
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.shape == Y.shape
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.shape[0]):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.shape[0]

def compute_gmm_score(model, samples_train, samples_test, gt_clusters_test):
    model.fit(samples_train)
    preds = model.predict(samples_test)
    return (
        adjusted_rand_score(gt_clusters_test, preds),
        normalized_mutual_info_score(gt_clusters_test, preds),
        cluster_acc(gt_clusters_test, preds),
        preds,
    )

def compute_embeded_gmm_score(model, test_loader):
    model.freeze()
    embeddings = []
    labels = []
    for x,gt, _ in test_loader:
        if model.hparams.device=='cuda':
            x = x.cuda()

        z, _ = model.pretrain_embed(x)
        embeddings.append(z)
        labels.append(gt)
    embeddings = torch.cat(embeddings, 0).detach().cpu().numpy()
    labels = torch.cat(labels, 0).detach().cpu().numpy()
    preds = model.gmm.predict(embeddings)
    return (
        adjusted_rand_score(labels, preds),
        normalized_mutual_info_score(labels, preds),
        cluster_acc(labels,preds),
        preds,
    )


def get_data(
    cfg,
    use_pca=False,
    use_numpy=False
):
    # Get mnist dataset
    if cfg.name == 'mnist':
        train_dataset, pretrain_dataset, test_dataset = get_mnist_data(cfg.data_root)
    elif cfg.name == 'fashion_mnist':
        train_dataset, pretrain_dataset, test_dataset = get_fashion_mnist_data(cfg.data_root)
    elif cfg.name == 'stl10':
        train_dataset, pretrain_dataset, test_dataset = get_stl10_data(cfg.data_root)
    else:
        raise ValueError(f"{cfg.name} is not a valid dataset.")

    # Generate set of PCA embeddings for plots
    if test_dataset[0][0].shape[-1] > 2 and cfg.name != 'stl10':
        if not use_pca:
            umap_path = os.path.join('.','pretrain', f'umap_{cfg.name}.npy')
            if os.path.exists(umap_path):
                emb_samples = np.load(umap_path)
            else:
                emb_samples = np.concatenate(
                    [x[0].reshape(1, -1).numpy() for x in test_dataset], axis=0
                )
                emb_samples = umap.UMAP(n_components=2).fit_transform(emb_samples)
                np.save(umap_path, emb_samples)
        else:
            pca_path = os.path.join('.','pretrain', f'pca_{cfg.name}.npy')
            if os.path.exists(pca_path):
                emb_samples = np.load(pca_path)
            else:
                emb_samples = np.concatenate(
                    [x[0].reshape(1, -1).numpy() for x in test_dataset], axis=0
                )
                emb_samples = PCA(n_components=2).fit_transform(emb_samples)
                np.save(pca_path, emb_samples)
    elif cfg.name == 'stl10':
        # No UMAP embedding for stl10
        emb_samples = np.zeros((len(test_dataset), 2))
    else:
        emb_samples = samples_test
    emb_samples = torch.tensor(emb_samples)
    if use_numpy:
        samples_train = np.concatenate(
            [x[0].reshape(1, -1).numpy() for x in train_dataset], axis=0
        )
        gt_clusters_train = np.array([x[1] for x in train_dataset])
        train_dataset = (samples_train, gt_clusters_train)
        samples_test = np.concatenate(
            [x[0].reshape(1, -1).numpy() for x in test_dataset], axis=0
        )
        gt_clusters_test = np.array([x[1] for x in test_dataset])
        test_dataset = (samples_test, gt_clusters_test)
    else:
        test_samples = torch.stack([x[0] for x in test_dataset])
        test_labels = torch.tensor([x[1] for x in test_dataset])
        test_dataset = torch.utils.data.TensorDataset(test_samples, test_labels,emb_samples)
    return pretrain_dataset, train_dataset, test_dataset, emb_samples


def get_dataloader(pretrain_dataset, train_dataset, test_dataset, emb_samples, batch_size, pretrain_batch_size):   
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_dataset, 
        batch_size=pretrain_batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, 
        drop_last=True, 
        num_workers=0,
    )
    return pretrain_loader, train_loader, val_loader

def init_model(samples, cfg, logger):
    if cfg.experiment.name=='gmm':
        if cfg.experiment.prior_init == 'random':
             model = GaussianMixture(
                 n_components=cfg.experiment.n_clusters_model,
                 covariance_type='diag',
                 init_params='random'
             )
        elif cfg.experiment.prior_init == 'zero':
            model = GaussianMixture(
                 n_components=cfg.experiment.n_clusters_model,
                 covariance_type='diag',
                 means_init=np.zeros((cfg.experiment.n_clusters_model, samples.shape[-1]))
             )
        elif cfg.experiment.prior_init == 'pretrain':
            model = GaussianMixture(
                 n_components=cfg.experiment.n_clusters_model,
                 covariance_type='diag',
                 init_params='k-means++'
             )
        else:
            raise ValueError(f"Initialization {cfg.experiment.prior_init} is invalid")
        return model
    args = [cfg.experiment.n_clusters_model]
    kwargs ={
        'lr':cfg.experiment.lr,
        'seed':cfg.seed,
        'resample':cfg.experiment.resample,
        'hard':cfg.experiment.hard,
        'final_temp':cfg.experiment.final_temp,
        'min_temp_step':cfg.experiment.min_temp_step,
        'init_temp':cfg.experiment.init_temp,
        'tau_schedule':cfg.experiment.tau_schedule,
        'device':cfg.device,
        'beta':cfg.experiment.beta,
        'gamma':cfg.experiment.gamma,
        'delta': cfg.experiment.delta,
        'batch_size': cfg.experiment.batch_size,
        'prior_init': cfg.experiment.prior_init,
        'inp_dim': cfg.dataset.inp_dim,
        'l_dim': cfg.experiment.l_dim if cfg.dataset.name != 'stl10' else cfg.dataset.l_dim,
        'intermediate_dim': cfg.experiment.intermediate_dim,
        'arch': cfg.dataset.name,
        'reinit_priors': cfg.experiment.reinit_priors,
    }
    
    if cfg.experiment.name in ['drpm_clustering', 'embeded_gmm']:
        model = VariationalDRPMClustering(
            *args,
            **kwargs
        )
    elif cfg.experiment.name=='vade':
        model = VaDE(
            cfg.experiment.n_clusters_model,
            vanilla_vade=cfg.experiment.vanilla_vade,
            **kwargs
        )
    else:
        raise ValueError(f"{cfg.experiment.name} is an invalid experiment configuration.")
    if cfg.experiment.name == 'embeded_gmm' or 'pretrain' in cfg.experiment.prior_init:
        model.init_weights(
            samples, 
            retrain_pretrain=cfg.experiment.retrain_pretrain,
            pretrain_dir=os.path.join('.','pretrain', cfg.dataset.name), 
            logger=logger,
            seed=cfg.seed
        )
    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: Config):
    pl.utilities.seed.seed_everything(cfg.seed)

    # Initialize logger
    wandb_logger = WandbLogger(
        name=cfg.logging.wandb_run_name,
        config=OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True
        ),
        project=cfg.logging.wandb_project_name,
        entity="drpm",
        log_model=cfg.logging.wandb_checkpoints,
        save_dir=cfg.logging.dir_logs,
        # settings=wandb.Settings(start_method="thread") # To make sure multiprocessing of wandb does not interfere with hydra sweep
    )
    
    # Initialize data
    pretrain_dataset, train_dataset, test_dataset, emb_samples = get_data(
        cfg.dataset,
        use_pca=cfg.logging.use_pca,
        use_numpy=cfg.experiment.name=='gmm'
    )
    
    if cfg.experiment.name != 'gmm':
        pretrain_loader, train_loader, val_loader = get_dataloader(
            pretrain_dataset, train_dataset, test_dataset, emb_samples, cfg.experiment.batch_size, pretrain_batch_size=1024 if cfg.dataset.name != 'stl10' else 256
        )

        # Init the model
        model = init_model(pretrain_loader, cfg, wandb_logger)
    else:
        # Init the model
        model = init_model(train_dataset[0], cfg, wandb_logger)
    
    if cfg.logging.wandb_watch:
        wandb_logger.watch(model, log="all", log_freq=cfg.logging.wandb_log_freq)

    # Log groundtruth
    if cfg.experiment.name != 'gmm':
        artist = plt.scatter(
            emb_samples.cpu()[:, 0], emb_samples.cpu()[:, 1], c=np.array([
                x[1] for x in test_dataset
            ])
        )
        wandb_logger.log_metrics({"gt": wandb.Image(artist)})
        plt.close()
    
    if 'gmm' in cfg.experiment.name:
        if cfg.experiment.name=='gmm':
            samples_train = train_dataset[0]
            samples_test, gt_clusters_test = test_dataset
            # Log Gaussian Mixture performance
            rand_score_gmm, nmi_gmm, acc_gmm, preds_gmm = compute_gmm_score(
                model, samples_train, samples_test, gt_clusters_test
            )
        elif cfg.experiment.name=='embeded_gmm':
            # Log Gaussian Mixture performance
            rand_score_gmm, nmi_gmm, acc_gmm, preds_gmm = compute_embeded_gmm_score(model, val_loader)
        else:
            raise ValueError(f"Invalid experiment {cfg.experiment.name}")
        idx = preds_gmm.argsort()
        artist = plt.scatter(
            emb_samples.cpu()[idx, 0], emb_samples.cpu()[idx, 1], c=preds_gmm[idx]
        )
        wandb_logger.log_metrics(
            {
                "final_scores/final_rand_score": rand_score_gmm,
                "final_scores/final_nmi": nmi_gmm,
                "final_scores/final_acc":acc_gmm,
                "preds_gmm": wandb.Image(artist),
            }
        )
        plt.close()
        return rand_score_gmm, nmi_gmm
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.experiment.checkpoint_metric, 
            mode="max", 
            filename="best_rand_score", 
            save_last=True
        )
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(
            max_epochs=cfg.experiment.epochs,
            gpus=1 if cfg.device == "cuda" else 0,
            logger=wandb_logger,
            log_every_n_steps=cfg.logging.wandb_log_freq,
            check_val_every_n_epoch=cfg.logging.val_freq,
            callbacks=[checkpoint_callback,lr_monitor_callback]
        )
        trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)
        model.logger.log_metrics({f"final_scores/final_rand_score":model.final_rand})
        model.logger.log_metrics({f"final_scores/final_nmi":model.final_nmi})
        model.logger.log_metrics({f"final_scores/final_acc":model.final_acc})
        return model.final_rand, model.final_nmi, model.final_acc


if __name__ == "__main__":
    run_experiment()
