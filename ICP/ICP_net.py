import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical, Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):

    def __init__(self, obs_dim, latent_dim=6):
        super(Encoder, self).__init__()
        self.en1 = nn.Linear(obs_dim, latent_dim)
        self.apply(weights_init_)

    def forward(self, inputs):
        x = self.en1(inputs)
        return x

    def get_weight(self, inputs):
        latent = self.forward(inputs)
        loss = F.mse_loss(latent, torch.zeros_like(
            latent), reduction='none').sum(dim=-1, keepdim=True)
        return loss, latent


class Multi_Encoder(nn.Module):

    def __init__(self, obs_dim, split_num, latent_dim=16):
        super(Multi_Encoder, self).__init__()
        self.split_num = split_num
        self.encoders = nn.ModuleList(
            [Encoder(obs_dim, latent_dim=latent_dim) for _ in range(split_num)])

    def forward(self, obs):
        loss_list, latent_list = [], []
        for i in range(self.split_num):
            loss, latent = self.encoders[i].get_weight(obs)
            loss_list.append(loss)
            latent_list.append(latent)

        loss_all = torch.cat(loss_list, dim=-1)
        latent_all = torch.stack(latent_list, dim=-1)
        return loss_all.squeeze(), latent_all.squeeze()

    def sample(self, obs):
        loss_all, latent_all = self.forward(obs)
        prob = torch.softmax(loss_all, dim=-1)
        category = prob.max(dim=-1)[1]

        category_reshape = category.unsqueeze(-1).expand_as(latent_all[..., 0])
        latent = torch.gather(
            latent_all, -1, category_reshape.unsqueeze(-1).long()).squeeze()
        return category, latent, prob

    def get_loss(self, obs, mask):
        loss_all, _ = self.forward(obs)
        pi_softmax = torch.softmax(loss_all, dim=-1)
        mask_expand = mask.expand_as(pi_softmax)

        L_k = (loss_all * mask_expand).sum() / (mask_expand.sum() + 1e-4)
        L_arg = ((loss_all * pi_softmax).sum(dim=-1).mean(dim=-1,
                 keepdim=True) * mask[:, :, 0]).sum() / (mask[:, :, 0].sum() + 1e-4)
        loss_total = L_k + L_arg

        return loss_total

    def to(self, device):
        self.encoders = self.encoders.to(device)
