import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import wandb
import os
import itertools
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from drpm.diff_rand_part_model import RandPart
from arch.ae_arch import Encoder, Decoder, ResnetFeatureExtractor
import torch.nn.functional as F
from torchvision.utils import make_grid
import math
import gc
from pathlib import Path
from arch.loss import *

class BaseVariationalClustering(pl.LightningModule):
    """
    Base class for variational partition clustering
    """
    def __init__(
        self,
        n_clusters,
        batch_size=256,
        intermediate_dim=2000,
        inp_dim=784,
        l_dim=10,
        lr=1e-4,
        seed=1,
        resample=True,
        hard=True,
        final_temp=0.5,
        min_temp_step=100000,
        init_temp=1,
        tau_schedule='exp',
        device="cuda",
        beta=1.,
        gamma=1.,
        delta=1.,
        prior_init='pretrain',
        reinit_priors=False,
        arch='mnist',
        **kwargs,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        
        # List to hold outputs from training
        self.training_step_outputs = []

        # Initialize random partition module
        self.randpart = RandPart(n_cluster=n_clusters,device=self.hparams.device)
        
        # Learn scale of scores separately
        self.score_scale = nn.Parameter(torch.tensor(0.))
        if 'pretrain' in self.hparams.prior_init:
            self.score_scale = nn.Parameter(torch.tensor(1.))
        
        # estimate mus and log sigmas
        self.estimate_mu=nn.Linear(self.hparams.intermediate_dim,self.hparams.l_dim)
        self.estimate_log_sigma=nn.Linear(self.hparams.intermediate_dim,self.hparams.l_dim)
        
        # Initialize prior gaussians of clusters
        self.prior_mu = torch.nn.parameter.Parameter(torch.randn(self.hparams.l_dim, self.hparams.n_clusters,1))            
        self.prior_log_std = torch.nn.parameter.Parameter(torch.zeros(self.hparams.l_dim, self.hparams.n_clusters,1))
                
        # initialize encoder/decoder
        if self.hparams.arch in ['mnist', 'fashion_mnist']:
            self.encoder = Encoder(input_dim=self.hparams.inp_dim, hid_dim=self.hparams.intermediate_dim)
            self.decoder = Decoder(input_dim=self.hparams.inp_dim, hid_dim=self.hparams.l_dim)
        elif self.hparams.arch == 'stl10':
            self.feature_extractor = ResnetFeatureExtractor()
            self.encoder = Encoder(input_dim=self.hparams.inp_dim, hid_dim=self.hparams.intermediate_dim)
            self.decoder = Decoder(input_dim=self.hparams.inp_dim, hid_dim=self.hparams.l_dim,sigmoid=False)
        else:
            raise ValueError(f"No architecture found for {self.hparams.arch}.")
        
        # Store final rand score and nmi
        self.register_buffer("final_rand", torch.tensor(0))
        self.register_buffer("final_nmi", torch.tensor(0))
        self.register_buffer("final_acc", torch.tensor(0))

    def pretrain_rec(self, dataloader, batch_size=64, epochs=50, path='./pretrain', retrain_pretrain=False, logger=lambda *x, **y: None):
        if  not os.path.exists(path) or retrain_pretrain:
            
            opti = torch.optim.Adam( 
                self.parameters()
            )

            print('Pretraining......')
            loss_func = nn.MSELoss()
            epoch_bar=tqdm(range(epochs))
            for epoch in epoch_bar:
                L=0
                for (x,_),_ in tqdm(dataloader, leave=False):
                    if self.hparams.device=='cuda':
                        x=x.cuda()
                    if self.hparams.arch == 'stl10':
                        x = self.feature_extractor(x).detach()
                    
                    z1, _ = self.pretrain_embed(x)
                    x_hat = self.decoder(z1)
                    loss = loss_func( x, x_hat)

                    L+= loss.item()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                fig, _, im = self.log_reconstructions(x.detach().cpu(), x_hat.detach().cpu())
                logger.log_metrics({
                    "pretrain/reconstructions": wandb.Image(im),
                    "pretrain/epoch": epoch,
                    "pretrain/rec_loss": L/len(dataloader)
                })
                plt.close(fig)
                # Save model every 16 epochs
                if epoch % 16 == 0:
                    torch.save(self.state_dict(), path)
            torch.save(self.state_dict(), path)
        else:
            weights = torch.load(path,map_location=torch.device(self.hparams.device))
            self.load_state_dict(weights, strict=False)
        return dataloader
    
    def pretrain_embed(self, x):
        z_intermediate = self.encoder(x)
        return self.estimate_mu(z_intermediate), self.estimate_log_sigma(z_intermediate)
    
    def pretrain_tcl(self, dataloader, epochs=200, path='./pretrain', retrain_pretrain=False, logger=lambda *x, **y: None):
        # Initialize classification layer for simclr pretraining
        self.projection_layer = nn.Sequential(
            nn.Linear(self.hparams.intermediate_dim, self.hparams.intermediate_dim),
            nn.GELU(),
            nn.Linear(self.hparams.intermediate_dim, self.hparams.l_dim)
        )
        self.cluster_projection_layer = nn.Sequential(
            nn.Linear(self.hparams.intermediate_dim, self.hparams.intermediate_dim),
            nn.GELU(),
            nn.Linear(self.hparams.intermediate_dim, self.hparams.n_clusters),
            nn.Softmax(dim=-1)
        )
        self.projection_layer.to(self.hparams.device)
        self.cluster_projection_layer.to(self.hparams.device)
        if  not os.path.exists(path) or retrain_pretrain:
            opti = torch.optim.Adam( 
                self.parameters()
            )

            print('Pretraining......')
            epoch_bar=tqdm(range(epochs))
            for epoch in epoch_bar:
                cluster_loss = ClusterLoss()
                instance_loss = InfoNCE
                L=0
                L_rec = 0
                L_tcl_cluster = 0
                L_tcl_instance = 0
                for i, batch in enumerate(tqdm(dataloader, leave=False)):
                    (x1,x2),_ = batch
                    opti.zero_grad()
                    if self.hparams.device=='cuda':
                        x1=x1.cuda()
                        x2=x2.cuda()

                    z1, c1, emb1 = self.tcl_embed(x1)
                    z2, c2, emb2 = self.tcl_embed(x2)
                    tcl_instance_loss = instance_loss(z1, z2) 
                    tcl_cluster_loss = cluster_loss(c1, c2)
                    x1_hat = self.decoder(emb1)
                    x2_hat = self.decoder(emb2)
                    rec_loss = 0.5*(F.mse_loss(x1, x1_hat, reduction='none') + F.mse_loss(x2, x2_hat, reduction='none'))
                    rec_loss = rec_loss.reshape(rec_loss.shape[0],-1).mean(dim=-1).sum()

                    loss = tcl_instance_loss + tcl_cluster_loss + rec_loss

                    L+=loss.item()
                    L_tcl_cluster+=tcl_cluster_loss.item()
                    L_tcl_instance+=tcl_instance_loss.item()
                    L_rec+=rec_loss.item()

                    loss.backward()
                    opti.step()
                fig, _, im = self.log_reconstructions(x1, x1_hat)
                logger.log_metrics({
                    "pretrain/reconstructions": wandb.Image(im),
                    "pretrain/epoch": epoch,
                    "pretrain/loss": L/len(dataloader),
                    "pretrain/tcl_instance_loss": L_tcl_instance/len(dataloader),
                    "pretrain/tcl_cluster_loss": L_tcl_cluster/len(dataloader),
                    "pretrain/rec_loss": L_rec/len(dataloader)
                })
                plt.close(fig)
                # Save model every 16 epochs
                if epoch % 16 == 0:
                    torch.save(self.state_dict(), path)
            torch.save(self.state_dict(), path)
        else:
            weights = torch.load(path,map_location=torch.device(self.hparams.device))
            self.load_state_dict(weights, strict=False)
        return dataloader
            
    def tcl_embed(self, x):
        h = self.encoder(x)
        return self.projection_layer(h), self.cluster_projection_layer(h), self.estimate_mu(h)
    
    def init_weights(self, dataloader, pretrain_dir='./pretrain', retrain_pretrain=False, logger=lambda *x, **y: None, seed=-1):
        """
        pretrain autoencoder to initialize Gaussian Mixture Components like in VaDE
        Code snippet from https://github.com/GuHongyang/VaDE-pytorch
        """
        self.to(self.hparams.device)
        Path(pretrain_dir).mkdir(parents=True, exist_ok=True)
        if self.hparams.prior_init=='pretrain' or self.hparams.arch == 'stl10':
            # Path for pretrained model
            path = os.path.join(pretrain_dir,f'pretrain_model_{self.hparams.n_clusters}_cluster_seed_{seed}.pk')
            self.pretrain_rec(
                dataloader,
                epochs=50,
                path=path,
                retrain_pretrain=retrain_pretrain,
                logger=logger
            )
        elif self.hparams.prior_init=='pretrain_tcl':
            # Path for pretrained model
            path = os.path.join(pretrain_dir,f'pretrain_tcl_model_{self.hparams.n_clusters}_cluster_seed_{seed}.pk')
            self.pretrain_tcl(
                dataloader,
                epochs=200,
                path=path,
                retrain_pretrain=retrain_pretrain,
                logger=logger
            )
            # Initialize sigmas of posterior distribution with exp(-2)
            nn.init.zeros_(self.estimate_log_sigma.weight)
            self.estimate_log_sigma.bias.data.fill_(-2)
        else:
            raise ValueError(f'Unknown pretrain strategy "{self.hparams.prior_init}".')
        
    def compute_rpm_parameters(self, mu, log_sigma):        
        # Closed form computation of log omega and scores from current embeddings
        with torch.no_grad():
            # Create prior and approximate posterior distributions
            prior_mu, prior_log_sigma = self.get_prior_params()
            prior_dist = torch.distributions.normal.Normal(prior_mu, prior_log_sigma.exp()+1e-6)
            dist = torch.distributions.normal.Normal(mu.unsqueeze(1), log_sigma.exp().unsqueeze(1)+1e-6)
            # Compute KL divergences and compute score
            kl_divs = torch.distributions.kl.kl_divergence(dist, prior_dist).sum(dim=-1)
            log_score = self.hparams.n_clusters - torch.argmin(kl_divs, dim=-1)[None]
            # Compute log omegas from cluster log probabilities
            if self.current_epoch==0 and 'pretrain' not in self.hparams.prior_init:
                # Set log omegas to a constant in first epoch since GMM not initialized
                log_omega = torch.log(torch.ones(1,self.hparams.n_clusters)/self.hparams.n_clusters).type_as(mu)
            else:
                cluster_log_probs = prior_dist.log_prob(mu.unsqueeze(1)).sum(dim=-1)
                log_omega = cluster_log_probs - torch.logsumexp(cluster_log_probs, dim=-1, keepdim=True)
                log_omega = torch.log(log_omega.exp().mean(dim=0, keepdim=True)+1e-6)
        log_score = log_score*10**self.score_scale
        
        return log_omega, log_score
        
    def embed(self, x):
        z_intermediate = self.encoder(x)
        mu = self.estimate_mu(z_intermediate)
        log_sigma = self.estimate_log_sigma(z_intermediate)
        
        # Compute log scores and log omegas from latent space
        log_omega, log_scores = self.compute_rpm_parameters(mu, log_sigma)
        return mu, log_sigma, log_omega, log_scores
    
    def reparametrize(self,mu, log_sigma):
        """
        Reparametrized sampling from gaussian
        """
        dist = torch.distributions.normal.Normal(
            mu,
            log_sigma.exp()+1e-6
        )
        return dist.rsample()
       
    def training_step(self, batch, batch_idx, optimizer_idx=0, log=True):
        # Compute current temperature for randpart model
        curr_temp = self.compute_current_temperature()
        
        # training_step defines the train loop.
        # it is independent of forward
        if self.hparams.arch != 'umap':
            x, _ = batch
            if self.hparams.arch == 'stl10':
                x = self.feature_extractor(x).detach()
            # Compute embedding parameters of current sample
            mu, log_sigma, log_omega, log_scores = self.embed(x)
        else:
            x, x_enc, _ = batch
            # Compute embedding parameters of current sample
            mu, log_sigma, log_omega, log_scores = self.embed(x_enc)
        
        # Save mu embeddings for processing after training
        self.training_step_outputs.append(mu.detach())
        
        # Store metrics of each mc sample
        mean_log_p_n, mean_log_p_Ps = [], []
        Ps_hats,n_hats = [],[]
        rec_losses = []
        z_kl_divs, Y_kl_divs = [], []
        zs = []
        
        
        # Get current assignment matrix
        assignments, n_hat, n_hat_shifted, n_hat_integer, Ps_hat, log_p_n, log_p_Ps, z, x_hat = self.resample_and_reconstruct(
            mu, 
            log_sigma, 
            log_omega, 
            log_scores,
            g_noise=False,
            hard=self.hparams.hard,
            tau=curr_temp,
        )
        mean_log_p_n.append(log_p_n.mean())
        mean_log_p_Ps.append(log_p_Ps.mean())
        Ps_hats.append(Ps_hat)
        n_hats.append(n_hat)
        zs.append(z)

        # Compute loss based on assignments and log probabilities
        rec_loss, z_kl_div, Y_kl_div = self.compute_loss(
            x, 
            x_hat, 
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
            log_metrics=log if batch_idx%200==0 else False,
        )
        rec_losses.append(rec_loss)
        z_kl_divs.append(z_kl_div)
        Y_kl_divs.append(Y_kl_div)
            
        # Cast lists to tensors
        mean_log_p_n = torch.stack(mean_log_p_n)
        mean_log_p_Ps = torch.stack(mean_log_p_Ps)
        rec_losses = torch.stack(rec_losses)
        z_kl_divs = torch.stack(z_kl_divs)
        Y_kl_divs = torch.stack(Y_kl_divs)
        
        # Aggregate loss
        rec_losses =  1./x.shape[0] * torch.sum(rec_losses)
        beta = self.compute_current_beta()
        z_kl_divs = beta/x.shape[0]* torch.sum(z_kl_divs)
        Y_kl_divs = 1./x.shape[0] * torch.sum(Y_kl_divs)
        kl_divs = z_kl_divs + Y_kl_divs
        loss = rec_losses + kl_divs
        
        # Logging loss
        if log:
            self.log("annealing/temperature", curr_temp)
            self.log('annealing/beta', beta)
            self.log("loss/train_loss", loss)
            self.log("probabilities/log_p_n", torch.mean(mean_log_p_n))
            self.log("probabilities/log_p_Ps", torch.mean(mean_log_p_Ps))
            self.log("loss/rec_loss", rec_losses)
            self.log("loss/kl_div", kl_divs)
            self.log("loss/z_kl_div", z_kl_divs)
            self.log("loss/Y_kl_div", Y_kl_divs)
            for i in range(self.hparams.n_clusters):
                self.log(f"log_omegas/log_omega_{i}", log_omega[0,i])
                self.log(f"n_hats/n_hat_{i}", n_hat_integer[i].squeeze())
        return loss
    
    def compute_current_temperature(self):
        """
        Compute temperature based on current step
        """
        min_temp_step = self.hparams.min_temp_step
        if self.hparams.tau_schedule =='linear':
            if self.global_step < min_temp_step:
                curr_temp = (
                    1 - self.global_step / min_temp_step
                ) * self.hparams.init_temp + (
                    self.global_step / min_temp_step
                ) * self.hparams.final_temp

            else:
                curr_temp = self.hparams.final_temp
        elif self.hparams.tau_schedule =='exp':
            final_temp = self.hparams.final_temp
            init_temp = self.hparams.init_temp
            rate = (math.log(final_temp) - math.log(init_temp))/float(min_temp_step)
            curr_temp = max(init_temp*math.exp(rate*self.global_step),final_temp)
        else:
            raise ValueError(f"tau_schedule should be one of 'linear' or 'exp', found {self.hparams.final_temp}")
        return curr_temp
    
    def compute_current_beta(self):
        """
        Compute beta based on current step
        """
        min_step = self.hparams.min_temp_step
        final_beta = self.hparams.beta
        init_beta = self.hparams.beta*0.1
        rate = (math.log(final_beta) - math.log(init_beta))/float(min_step)
        curr_beta = min(init_beta*math.exp(rate*self.global_step),final_beta)
        return curr_beta
    
    def compute_current_gamma(self):
        """
        Compute gamma based on current step
        """
        return self.hparams.gamma
    
    def compute_current_delta(self):
        """
        Compute delta based on current step
        """
        return self.hparams.delta
    
    def compute_loss(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx, log_metrics=True):
        # After every epoch, log current rand_score
        if self.hparams.arch != 'umap':
            x, gt, emb_x = batch
            if self.hparams.arch == 'stl10':
                x = self.feature_extractor(x).detach()
            # Compute embedding parameters of current sample
            mu, log_sigma, log_omega, log_scores = self.embed(x)
        else:
            x, x_enc, gt, emb_x = batch
            # Compute embedding parameters of current sample
            mu, log_sigma, log_omega, log_scores = self.embed(x_enc)
        
        preds, n_hat, n_hat_shifted, n_hat_integer, Ps_hat, log_p_n, log_p_Ps, z, x_hat = self.resample_and_reconstruct(
            mu, 
            log_sigma, 
            log_omega, 
            log_scores,
            g_noise=False,
            hard=True,
            reparametrize_z=False
        )

        # Initaialize logger dict
        log = {}
        
        # Compute rand score and nmi with max probability
        preds_cluster_prob = self.get_preds_cluster_prob(mu)

        # Log current clustering of first validation batch
        emb_x = emb_x.cpu()
        visualization_points = [emb_x[:, 0], emb_x[:, 1]]

        # Log histogram of current log_scores
        log["parameters/log_score"] = wandb.Histogram(log_scores.detach().cpu())
        log["parameters/log_omega"] = wandb.Histogram(log_omega.detach().cpu())
        log["parameters/mus"] = wandb.Histogram(mu.detach().cpu())
        log["parameters/log_sigmas"] = wandb.Histogram(log_sigma.detach().cpu())
        
        # Log reconstructions for first batch
        if batch_idx == 0:
            fig, _, im = self.log_reconstructions(x, x_hat)
            if log_metrics:
                self.logger.log_metrics({"reconstructions": wandb.Image(im)})
            plt.close(fig)
        # Log validation metrics
        if log_metrics:
            self.logger.log_metrics(log)

        # return score_cluster_prob, score_partition, artist_partition
        return preds_cluster_prob, gt.cpu(), visualization_points
    
    def validation_epoch_end(self, out):
        if self.current_epoch==0:
            return
        # Clear plot
        plt.cla()
        plt.clf()
        
        # Aggregate predictions
        x,y, preds_cluster_prob, gt = [],[],[],[]
        for curr_pred_cluster_prob, curr_gt, (curr_x, curr_y) in out:
            x.append(curr_x)
            y.append(curr_y)
            preds_cluster_prob.append(curr_pred_cluster_prob)
            gt.append(curr_gt)
        x = torch.cat(x)
        y = torch.cat(y)
        preds_cluster_prob = torch.cat(preds_cluster_prob)
        gt = torch.cat(gt)
        
        # Sort predictions by index to make visualization prettier
        idx = preds_cluster_prob.argsort()
        preds = preds_cluster_prob[idx]
        plt_x = x[idx]
        plt_y = y[idx]
        # Log visualization
        fig, ax = plt.subplots()
        scat = ax.scatter(plt_x,plt_y,c=preds)
        fig.tight_layout()
        self.logger.log_metrics({"val_preds_cluster_prob": wandb.Image(scat)})
        plt.close(fig)
        
        # log cluster probability prediction scores
        score_cluster_prob = adjusted_rand_score(gt.cpu(), preds_cluster_prob)
        nmi_cluster_prob = normalized_mutual_info_score(gt, preds_cluster_prob)
        acc_cluster_prob = self.cluster_acc(gt, preds_cluster_prob)
        self.log("validation_score/nmi_cluster_prob", nmi_cluster_prob)
        self.log("validation_score/rand_score_cluster_prob", score_cluster_prob)
        self.log("validation_score/acc_cluster_prob", acc_cluster_prob)
                
        # Save current final scors
        self.final_rand = torch.tensor(score_cluster_prob)
        self.final_nmi = torch.tensor(nmi_cluster_prob)
        self.final_acc = torch.tensor(acc_cluster_prob)
    
    def log_reconstructions(self, x, x_hat):
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots()
        if len(x.shape)==2:
            if self.hparams.arch == 'stl10':
                # For STL-10 just plot latent vector (2048 dim) as image
                x = x.reshape(-1,1,32,64)
                x_hat = x_hat.reshape(-1,1,32,64)
            else:
                resolution = int(math.sqrt(x.shape[-1]))
                x = x.reshape(-1,1,resolution,resolution)
                x_hat = x_hat.reshape(-1,1,resolution,resolution)
            recs = make_grid(torch.cat([x[:64],x_hat[:64]],dim = -1).cpu()).permute(1,2,0)
            #Scale to [0,1]
            recs = (recs - recs.min())/(recs.max()-recs.min())
            im = ax.imshow(recs,cmap='gray')
        elif len(x.shape)==4:
            recs = make_grid(torch.cat([x[:64],x_hat[:64]],dim = -1).cpu()).permute(1,2,0)
            im = ax.imshow(recs)
        fig.tight_layout()
        return fig, ax, im
    
    def cluster_acc(self, Y, Y_pred):
        """
        From https://github.com/GuHongyang/VaDE-pytorch
        """
        from scipy.optimize import linear_sum_assignment
        assert Y_pred.size() == Y.size()
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size()[0]):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)
        return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size()[0]
    
    def configure_optimizers(self):
        raise NotImplementedError()