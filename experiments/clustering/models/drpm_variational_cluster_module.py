import torch
import copy
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import rand_score, normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder
import wandb
import os
import itertools
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from drpm.diff_rand_part_model import RandPart
import torch.nn.functional as F
from torchvision.utils import make_grid
from models.drpm_base_variational_cluster_module import BaseVariationalClustering
from drpm.pl import PL
import numpy as np
    
# define the LightningModule
class VariationalDRPMClustering(BaseVariationalClustering):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # GMM for priors        
        self.gmm = GaussianMixture(
            n_components=self.hparams.n_clusters, 
            covariance_type='diag', 
            init_params='k-means++',
            n_init=10,
            reg_covar=1e-4
        )
        
        # set prior cluster distributions, properly initialize them in init_weights
        self.prior_log_omega = torch.nn.parameter.Parameter(torch.log(torch.ones(self.hparams.n_clusters)/self.hparams.n_clusters))
    
    def initialize_gaussian_priors(self, embs):
        self.gmm.fit(embs)
        preds = self.gmm.predict (embs)
        preds = self.gmm.predict_proba(embs)

        self.prior_log_omega = torch.nn.parameter.Parameter(
            torch.log(torch.from_numpy(self.gmm.weights_).float().to(self.hparams.device))
        )
        self.prior_mu = torch.nn.parameter.Parameter(
            torch.from_numpy(self.gmm.means_).float().permute(1,0).unsqueeze(-1).to(self.hparams.device)
        )
        self.prior_log_std = torch.nn.parameter.Parameter(
            0.5*torch.log(torch.from_numpy(self.gmm.covariances_).float().permute(1,0).unsqueeze(-1)).to(self.hparams.device)
        )
    
    def init_weights(self, dataloader, **kwargs):
        super().init_weights(
            dataloader,
            **kwargs
        )
        Z = []
        indexes = []
        with torch.no_grad():
            for (x,_),_ in dataloader:
                if self.hparams.device=='cuda':
                    x = x.cuda()
                if self.hparams.arch == 'stl10':
                    x = self.feature_extractor(x).detach()

                z1, _ = self.pretrain_embed(x)
                Z.append(z1)

        Z = torch.cat(Z, 0).detach().cpu().numpy()
        self.initialize_gaussian_priors(Z)
    
    def on_train_epoch_end(self, losses):
        if self.hparams.reinit_priors and self.current_epoch%10==0 and self.global_step<self.hparams.min_temp_step:
            embs = torch.cat(self.training_step_outputs).cpu().numpy()
            self.initialize_gaussian_priors(embs)
        self.training_step_outputs.clear()
            
    def get_prior_params(self):
        prior_mu = self.prior_mu.permute(2,1,0)
        prior_log_sigma = self.prior_log_std.permute(2,1,0)
        return prior_mu, prior_log_sigma
            
    def get_log_omega(self):
        normalized_log_omega = self.prior_log_omega-torch.logsumexp(self.prior_log_omega,dim=0)
        return normalized_log_omega
    
    def compute_kl_div(self, mu, log_sigma, normalize=False):
        dist = torch.distributions.normal.Normal(
            mu.unsqueeze(1),
            log_sigma.exp().unsqueeze(1)+1e-6
        )
        prior_mu, prior_log_sigma = self.get_prior_params()
        prior_dist = torch.distributions.normal.Normal(
            prior_mu,
            prior_log_sigma.exp()+1e-6
        )
        kl_div = torch.distributions.kl.kl_divergence(dist, prior_dist)
        if normalize:
            kl_div = F.normalize(kl_div, dim=(1,2), p=1)
        return kl_div
        
    
    def compute_log_soft_assignments(self, z):
        det=1e-10
        log_pi = self.get_log_omega().unsqueeze(0)
        mu, log_sigma = self.get_prior_params()
        dist = torch.distributions.normal.Normal(mu, log_sigma.exp()+1e-6)
        z = z.unsqueeze(1)
        yita_c=log_pi+dist.log_prob(z).sum(-1)+det
        yita_c=yita_c-torch.logsumexp(yita_c,dim=1,keepdims=True)#batch_size*Clusters
        if not torch.all(torch.abs(yita_c.exp().sum(-1)-1)<1e-5):
            print(f"soft assignments not adding up to one: {yita_c.exp().sum(-1)}")
        return yita_c
    
    def compute_resample_kl_divs(
        self, 
        z,
        mu,
        log_sigma,
        log_scores,
        log_omega,
        assignments, 
        n_hat, 
        n_hat_shifted,
        n_hat_integer,
        Ps_hat, 
        log_p_n,
        log_p_Ps,
    ):
        log = {}
        """
        Compute z  KL divergence
        """
        prior_mu, prior_log_sigma = self.get_prior_params()
        prior_dist = torch.distributions.normal.Normal(prior_mu, prior_log_sigma.exp()+1e-6)
        dist = torch.distributions.normal.Normal(mu.unsqueeze(1), log_sigma.exp().unsqueeze(1)+1e-6)
        
        n_mc_samples = 100
        n_hat_shifted = n_hat_shifted.repeat(n_mc_samples,1,1)
        log_scores_rep = log_scores.repeat(n_mc_samples,1)
        perms, assignments, log_p_Ps = self.randpart.get_partition(
            log_scores_rep, 
            n_hat_shifted, 
            self.compute_current_temperature(),
            self.hparams.resample,
            self.hparams.hard
        )
        z_kl_div = (assignments*torch.distributions.kl.kl_divergence(dist, prior_dist).sum(dim=-1).T[None]).sum()/n_mc_samples
        
        log["parameters/prior_mu"] = wandb.Histogram(prior_mu.detach().cpu())
        log["parameters/prior_log_sigma"] = wandb.Histogram(prior_log_sigma.detach().cpu())
        log["parameters/prior_log_omega"] = wandb.Histogram(self.get_log_omega().detach().cpu())
        log["parameters/log_score"] = wandb.Histogram(log_scores.detach().cpu())
        log["parameters/log_omega"] = wandb.Histogram(log_omega.detach().cpu())
        
        """
        Compute partition KL divergence
        """
        
        # Compute partition probability
        
        ## Create integer number from mvhg sample to compute Pi_Y
        n_hat_integer_cat = torch.cat(n_hat,dim=1).squeeze(-1).float()
        n_hat_integer_cat = n_hat_integer_cat*torch.arange(z.shape[0]+1).unsqueeze(0).type_as(z)
        n_hat_integer_cat = n_hat_integer_cat.sum(dim=-1)
        n_hat_integer = [n_hat_integer_cat[:,i].reshape(-1,1,1) for i in range(n_hat_integer_cat.shape[-1])]
        
        ## Compute number of possible permutations for current mvhg sample
        log_num_Y_permutations = (n_hat_integer_cat + 1).lgamma().sum(dim=-1)
        
        ## Get maximum probability of permutations by ignoring gumbel noise in PL sampling
        sort = PL(
            log_scores, tau=self.compute_current_temperature(), g_noise=False
        )
        max_log_p_Ps = sort.log_prob(sort.rsample([1]))
        
        ## Log computed upper bound on partition probability
        self.log("probabilities/upper_bound_log_p_Y", (log_num_Y_permutations+log_p_n).mean()+max_log_p_Ps)
        
        ## Compute mvhg prior probability
        prior_log_omega = self.get_log_omega().unsqueeze(0)
        prior_log_p_n = self.randpart.get_log_prob_mvhg(
            prior_log_omega.repeat(n_mc_samples,1),
            n_hat_integer,
            n_hat,
        )
        
        ## Get maximum prior probability of permutations by ignoring gumbel noise in PL sampling
        sort = PL(
            torch.zeros_like(log_scores), tau=self.compute_current_temperature(), g_noise=False
        )
        prior_log_p_Ps = sort.log_prob(perms).mean()
        
        ## Compute kl div of partition
        gamma = self.compute_current_gamma()
        delta = self.compute_current_delta()
        Y_kl_div = gamma*torch.relu((log_p_n+log_num_Y_permutations-prior_log_p_n).mean())+delta*(max_log_p_Ps-prior_log_p_Ps)/z.shape[0]
        
        # Log wandb metrics
        self.log('annealing/gamma', gamma)
        self.log('annealing/delta', delta)
        
        return z_kl_div, Y_kl_div, log
    
    def compute_deterministic_kl_divs(
        self, 
        z,
        mu,
        log_sigma,
        log_scores,
        log_omega,
        assignments, 
        n_hat, 
        n_hat_shifted,
        n_hat_integer,
        Ps_hat, 
        log_p_n,
        log_p_Ps,
    ):
        log = {}
        """
        Compute z  KL divergence
        """
        prior_mu, prior_log_sigma = self.get_prior_params()
        prior_dist = torch.distributions.normal.Normal(prior_mu, prior_log_sigma.exp()+1e-6)
        dist = torch.distributions.normal.Normal(mu.unsqueeze(1), log_sigma.exp().unsqueeze(1)+1e-6)
                
        z_kl_div = (assignments.permute(0,2,1)*torch.distributions.kl.kl_divergence(dist, prior_dist).sum(dim=-1).unsqueeze(0)).sum()
        
        log["parameters/prior_mu"] = wandb.Histogram(prior_mu.detach().cpu())
        log["parameters/prior_log_sigma"] = wandb.Histogram(prior_log_sigma.detach().cpu())
        log["parameters/prior_log_omega"] = wandb.Histogram(self.get_log_omega().detach().cpu())
        log["parameters/log_score"] = wandb.Histogram(log_scores.detach().cpu())
        log["parameters/log_omega"] = wandb.Histogram(log_omega.detach().cpu())
        
        """
        Compute partition KL divergence
        """
        
        # Compute partition probability
        
        ## Create integer number from mvhg sample to compute Pi_Y
        n_hat_integer_cat = torch.cat(n_hat,dim=1).squeeze(-1).float()
        n_hat_integer_cat = n_hat_integer_cat*torch.arange(z.shape[0]+1).unsqueeze(0).type_as(z)
        n_hat_integer_cat = n_hat_integer_cat.sum(dim=-1)
        n_hat_integer = [n_hat_integer_cat[:,i].reshape(-1,1,1) for i in range(n_hat_integer_cat.shape[-1])]
        
        ## Compute number of possible permutations for current mvhg sample
        log_num_Y_permutations = (n_hat_integer_cat + 1).lgamma().sum()
        
        ## Compute mvhg prior probability
        prior_log_omega = self.get_log_omega().unsqueeze(0)
        prior_log_p_n = self.randpart.get_log_prob_mvhg(
            prior_log_omega,
            n_hat_integer,
            n_hat,
        ).squeeze()
        
        ## Get maximum prior probability of permutations by ignoring gumbel noise in PL sampling
        sort = PL(
            torch.zeros_like(log_scores), tau=self.compute_current_temperature(), g_noise=False
        )
        prior_log_p_Ps = sort.log_prob(Ps_hat)
        
        ## Compute kl div of partition
        gamma = self.compute_current_gamma()
        delta = self.compute_current_delta()
        Y_kl_div = -gamma*(prior_log_p_n + prior_log_p_Ps/z.shape[0])
        ## Add PI_Y as an additional regularizer for the log omegas
        Y_kl_div += gamma*log_num_Y_permutations
        
        # Log wandb metrics
        self.log('annealing/gamma', gamma)
        
        return z_kl_div, Y_kl_div, log
    
    def compute_loss(
        self,
        x,
        x_hat,
        *args,
        log_metrics=True
    ):
        """
        Compute reconstruction loss
        """
        if self.hparams.arch=='stl10':
            # Latent vectors of stl-10 are not in [0,1] so we need to use mse_loss
            rec_loss = F.mse_loss(x_hat,x, reduction='sum')
        else:
            rec_loss = F.binary_cross_entropy(x_hat,x, reduction='sum')
        
        """
        Compute KL Divergences
        """
        if self.hparams.resample:
            z_kl_div, Y_kl_div, log = self.compute_resample_kl_divs(*args)
        else:
            z_kl_div, Y_kl_div, log = self.compute_deterministic_kl_divs(*args)
            
        if log_metrics:
            self.logger.log_metrics(log)
            
        # Return loss terms
        return rec_loss, z_kl_div, Y_kl_div
    
    def sample_mvhg(self, log_omega, partition_size, num_samples, hard=True):
        m = torch.tensor([partition_size for i in range(log_omega.shape[-1])]).type_as(log_omega)
        n = (torch.tensor(partition_size).unsqueeze(0).repeat(num_samples, 1).type_as(log_omega)
        )
        ohts, num_per_cluster, filled_ohts, log_p = self.randpart.mvhg(
            m, n, log_omega.repeat(num_samples,1), self.compute_current_temperature(), add_noise=True, hard=hard
        )
        return ohts, num_per_cluster, log_p
        
    def reparametrize(self, mu, log_sigma, deterministic=False):
        if deterministic:
            return mu
        return super().reparametrize(mu,log_sigma)
        
    def get_assigned_parameters(self, mu, log_sigma, assignments):
        mu = (assignments*mu).sum(dim=1).permute(1,0)
        log_sigma = (assignments*log_sigma).sum(dim=1).permute(1,0)
        return mu, log_sigma
    
    def resample_and_reconstruct(
        self,
        mu, 
        log_sigma, 
        log_omega, 
        log_scores,
        g_noise=False,
        hard=True,
        reparametrize_z=True,
        tau=None
    ):
        # Set temperature to be the default if None
        if tau is None:
            tau=self.hparams.final_temp
        assignment_valid = False
        retries = 0
        while not assignment_valid:
            retries += 1
            assignments, Ps_hat, n_hat, n_hat_shifted, n_hat_integer, log_p_n, log_p_Ps = self.randpart(
                log_scores,
                log_omega,
                g_noise=g_noise,
                hard_pi=hard,
                temperature=tau,
            )
            # Ensure assignment is valid in case of rounding error in resampling
            if torch.all(assignments.sum(dim=1)==1) or not hard or retries >= 5:
                assignment_valid = True
            elif retries >=2:
                print("Sampled invalid partition matrix, retrying...")
        # compute reconstruction given cluster assignments
        z = self.reparametrize(mu, log_sigma, deterministic=not reparametrize_z)
        x_hat = self.decoder(z)
        return assignments, n_hat, n_hat_shifted, n_hat_integer, Ps_hat, log_p_n, log_p_Ps, z, x_hat
    
    def get_kl_div_preds(
        self,
        mu,
        log_sigma
    ):
        kl_div = self.compute_kl_div(mu, log_sigma).sum(dim=-1)
        preds = kl_div.argmin(dim=1).cpu()
        return preds
        
    
    def get_preds_cluster_prob(
        self,
        z
    ):
        prior_mu, prior_log_std = self.get_prior_params()
        dist = torch.distributions.normal.Normal(
            prior_mu,
            prior_log_std.exp()+1e-6
        )
        z_cluster_probs = dist.log_prob(z.unsqueeze(1)).sum(dim=-1)
        preds = z_cluster_probs.argmax(dim=1).cpu()
        return preds
    
    def configure_optimizers(self):
        # Create parameter groups and schedulers
        params = [
            {
                'params': itertools.chain(
                    self.decoder.parameters(),
                    self.estimate_mu.parameters(),
                    self.estimate_log_sigma.parameters(),
                    nn.ParameterList([
                        self.score_scale, 
                    ])
                ),
            },
        ]
        lr_decay_lambdas=[
            lambda x: 1,
        ]
        # Freeze encoder if using pretraining in the DRPM
        if 'pretrain' not in self.hparams.prior_init:
            params.append({
                'params': self.encoder.parameters(),
            })
            lr_decay_lambdas.append(lambda x: 0.99*x)
        
        # Add priors as parameter group
        params.append(
            {
                'params': nn.ParameterList([
                    self.prior_mu, 
                    self.prior_log_std,
                    self.prior_log_omega
                ]),
                'lr': self.hparams.lr,
            })
        lr_decay_lambdas.append(lambda x: 1.)
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_decay_lambdas
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}