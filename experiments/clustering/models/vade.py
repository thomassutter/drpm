import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import os
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from torchvision.utils import make_grid
import wandb
from models.drpm_base_variational_cluster_module import BaseVariationalClustering


class VaDE(BaseVariationalClustering):
    def __init__(
        self,
        n_clusters,
        **kwargs
    ):
        super().__init__(n_clusters, **kwargs)
        self.pi_=nn.Parameter(torch.log(torch.FloatTensor(n_clusters,).fill_(1)/n_clusters),requires_grad=True)
        self.mu_c = nn.Parameter(self.prior_mu)
        self.log_sigma2_c=nn.Parameter(self.prior_log_std)
        # GMM for priors        
        self.gmm = GaussianMixture(
            n_components=self.hparams.n_clusters, 
            covariance_type='diag', 
            init_params='k-means++',
            n_init=10,
            reg_covar=1e-5
        )



    def init_weights(self, dataloader, **kwargs):
        super().init_weights(
            dataloader,
            **kwargs
        )
        # Initialize estimate_log_sigma weights with estimate_mu weights according to
        # https://github.com/GuHongyang/VaDE-pytorch
        self.estimate_log_sigma.load_state_dict(self.estimate_mu.state_dict())
        Z = []
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

    def embed(self, x):
        z = self.encoder(x)
        return self.estimate_mu(z), self.estimate_log_sigma(z)
    
    def get_prior_params(self):
        if 'pretrain' in self.hparams.prior_init:
            return self.mu_c, self.log_sigma2_c
        prior_mu = self.mu_c.permute(2,1,0).squeeze(0)
        prior_log_sigma = self.log_sigma2_c.permute(2,1,0).squeeze(0)
        return prior_mu, prior_log_sigma
    
    def get_pi(self):
        normalized_log_pi = self.pi_-self.pi_.logsumexp(dim=-1,keepdim=True)
        return normalized_log_pi.exp()
    
    def validation_step(self, batch, batch_idx, log_metrics=True):
        # After every epoch, log current rand_score
        x, gt, emb_x = batch
        if self.hparams.arch == 'stl10':
            x = self.feature_extractor(x).detach()

        # Compute predictions
        preds_cluster_prob = self.predict(x)
        
        # Log current clustering of first validation batch
        emb_x = emb_x.cpu()
        visualization_points = [emb_x[:, 0], emb_x[:, 1]]
        
        # Log reconstructions for first batch
        log = {}
        if batch_idx == 0:
            x_hat = self.decoder(self.embed(x)[0])
            fig, _, im = self.log_reconstructions(x, x_hat)
            if log_metrics:
                self.logger.log_metrics({"reconstructions": wandb.Image(im)})
            plt.close(fig)

        # return score_cluster_prob, score_partition, artist_partition
        return preds_cluster_prob, gt.cpu(), visualization_points

    def predict(self,x):
        z_mu, z_sigma2_log = self.embed(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.get_pi()
        mu_c, log_sigma2_c = self.get_prior_params()
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)+1e-6)+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu()
        return torch.argmax(yita,dim=1)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0, log=True):
        x, _ = batch
        if self.hparams.arch == 'stl10':
            x = self.feature_extractor(x).detach()
        loss = self.ELBO_Loss(x)
        if log:
            self.log("train_loss", loss)
        return loss


    def ELBO_Loss(self,x,L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = self.embed(x)
        self.training_step_outputs.append(z_mu.detach())
        for l in range(L):

            z=self.reparametrize(z_mu,z_sigma2_log/2)

            x_pro=self.decoder(z)
            
            if self.hparams.arch=='stl10':
                # Latent vectors of stl-10 are not in [0,1] so we need to use mse_loss
                L_rec+=F.mse_loss(x_pro,x)
            else:
                L_rec+=F.binary_cross_entropy(x_pro,x)

        L_rec/=L

        Loss=L_rec*x.size(1)

        pi=self.get_pi()
        mu_c, log_sigma2_c = self.get_prior_params()

        z = self.reparametrize(z_mu,z_sigma2_log/2)
        yita_c=torch.exp(torch.log(pi.unsqueeze(0)+det)+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita_c=yita_c/(yita_c.sum(1).view(-1,1))+det#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/(torch.exp(log_sigma2_c.unsqueeze(0))+det),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)+det),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return Loss

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.hparams.n_clusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)   
    
    def configure_optimizers(self):
        if self.hparams.vanilla_vade:
            optimizer=Adam(self.parameters(),lr=2e-3)
            scheduler=StepLR(optimizer,step_size=10,gamma=0.95)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        # Create parameter groups
        params = itertools.chain(
                    self.decoder.parameters(),
                    self.estimate_mu.parameters(),
                    self.estimate_log_sigma.parameters(),
                    nn.ParameterList([
                        self.mu_c, 
                        self.log_sigma2_c,
                        self.pi_
                    ])
                )
        # Freeze encoder if using pretraining in the DRPM
        if 'pretrain' not in self.hparams.prior_init:
            params =itertools.chain(params, self.encoder.parameters())

        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        
        return {'optimizer': optimizer}
    
    def on_train_epoch_end(self, losses):
        if self.hparams.reinit_priors and self.current_epoch%10==0 and self.global_step<self.hparams.min_temp_step:
            embs = torch.cat(self.training_step_outputs).cpu().numpy()
            self.initialize_gaussian_priors(embs)
        self.training_step_outputs.clear()

    def initialize_gaussian_priors(self, embs):
        self.gmm.fit(embs)
        preds = self.gmm.predict (embs)
        preds = self.gmm.predict_proba(embs)

        self.pi_.data = torch.log(torch.from_numpy(self.gmm.weights_).float().type_as(self.mu_c))
        self.mu_c.data = torch.from_numpy(self.gmm.means_).float().type_as(self.mu_c)
        self.log_sigma2_c.data = torch.log(torch.from_numpy(self.gmm.covariances_).float()).type_as(self.mu_c)
        
    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/(torch.exp(log_sigma2)+1e-6),1))