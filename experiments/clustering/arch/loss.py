import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def InfoNCE(
    anchor_rec: torch.Tensor, 
    positive_rec: torch.Tensor,
    tau: float = 0.5, 
    symetric: bool =True,) -> torch.Tensor:
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    sim11 = sim_metric(anchor_rec.unsqueeze(-2), anchor_rec.unsqueeze(-3)) / tau
    sim22 = sim_metric(positive_rec.unsqueeze(-2), positive_rec.unsqueeze(-3)) / tau
    sim12 = sim_metric(anchor_rec.unsqueeze(-2), positive_rec.unsqueeze(-3))/ tau
    # removal of 1:1 pairs
    sim11 = sim11.flatten()[1:].view(sim11.shape[0]-1, sim11.shape[0]+1)[:,:-1].reshape(sim11.shape[0], sim11.shape[0]-1)
    sim22 = sim22.flatten()[1:].view(sim22.shape[0]-1, sim22.shape[0]+1)[:,:-1].reshape(sim22.shape[0], sim22.shape[0]-1)
    d = sim12.shape[-1]
    raw_scores1 = torch.cat([sim12, sim11], dim=-1)
    targets1 = torch.arange(d, dtype=torch.long, device=raw_scores1.device)
    total_loss_value = torch.nn.CrossEntropyLoss()(raw_scores1, targets1)
    if symetric: 
        sim12 = sim_metric(positive_rec.unsqueeze(-2) ,
                           anchor_rec.unsqueeze(-3))/ tau
        # creating matrix with all similarities
        raw_scores1 = torch.cat([sim12, sim22], dim=-1)
        total_loss_value += torch.nn.CrossEntropyLoss()(raw_scores1,targets1)
        total_loss_value *= 0.5
    losses_value = total_loss_value
    return losses_value

class ClusterLoss(nn.Module):
    """
    Instance and cluster losses from twin contrastive learning
    """

    LARGE_NUMBER = 1e4

    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, c_aug0, c_aug1):
        p_i = c_aug0.sum(0).view(-1)
        p_i /= p_i.sum()
        en_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_aug1.sum(0).view(-1)
        p_j /= p_j.sum()
        en_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        en_loss = en_i + en_j
        
        loss = InfoNCE(c_aug0.T, c_aug1.T, tau=self.tau)

        return loss + en_loss