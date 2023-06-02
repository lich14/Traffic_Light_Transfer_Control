import os
from traceback import print_tb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .util import init
from .popart import PopArt
from torch.distributions import Categorical, Normal
from ICP.ICP_net import Multi_Encoder


def check(input):
    output = input if type(
        input) == np.ndarray else input.detach().to('cpu').numpy()
    return output


class Agent(nn.Module):
    def __init__(self, args, obs_dim, n_agents, comm_dim, hidden_size, category_num, latent_dim, device=torch.device("cpu")):
        super(Agent, self).__init__()
        self.args = args
        active_func = nn.ReLU()
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain('relu')
        lane_info_dim = (obs_dim - 2) // 4

        self.category_num = category_num
        self.ICP = Multi_Encoder(
            obs_dim, category_num, latent_dim)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.comm_mlp = nn.Sequential(
            init_(nn.Linear(lane_info_dim, 32)), active_func)

        self.comm_gru = nn.GRU(
            input_size=32,
            num_layers=1,
            hidden_size=comm_dim,
            batch_first=True,
        )

        for name, param in self.comm_gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(latent_dim, hidden_size)), active_func)

        self.rnn = nn.GRU(
            input_size=hidden_size,
            num_layers=1,
            hidden_size=hidden_size,
            batch_first=True,
        )

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.actor_fc2 = nn.ModuleList(
            [init_(nn.Linear(hidden_size + 4 * comm_dim, 1)) for _ in range(category_num)])

        self.critic = nn.Sequential(
            init_(nn.Linear(obs_dim * n_agents, hidden_size)),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            init_(nn.Linear(hidden_size, 1))
        )

        log_std = torch.zeros([category_num, 1])
        self.log_std = torch.nn.Parameter(log_std)

        self.params = list(self.fc1.parameters()) + \
            list(self.rnn.parameters()) + \
            list(self.actor_fc2.parameters()) + \
            list(self.critic.parameters()) + \
            list(self.ICP.parameters()) + \
            list(self.comm_mlp.parameters()) + \
            list(self.comm_gru.parameters()) + \
            [self.log_std]

        self.to(device)
        self.ICP.to(device)
        self.save_path = f'checkpoints_PPO_ICP/points_{args.points}_difficult_{args.difficult}/category_{args.category_num}_reward_para_{args.reward_para}_seed_{args.seed}'

    def comm_n2s(self, obs, m):
        h = self.comm_mlp(obs)
        return self.comm_gru(h, m)[0]

    def comm_s2n(self, obs, m):
        h = self.comm_mlp(obs)
        return self.comm_gru(h, m)[0]

    def comm_w2e(self, obs, m):
        h = self.comm_mlp(obs)
        return self.comm_gru(h, m)[0]

    def comm_e2w(self, obs, m):
        h = self.comm_mlp(obs)
        return self.comm_gru(h, m)[0]

    def sample(self, obs, state, m_N2S, m_S2N, m_W2E, m_E2W, hidden_states, deterministic=False):

        category, latent, _ = self.ICP.sample(obs)

        x = self.fc1(latent)
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1)

        features, hidden_states = self.rnn(x, hidden_states)
        features = features.squeeze().reshape(
            obs.shape[:2] + (features.shape[-1],))

        features_add_comm = torch.cat(
            [features, m_N2S, m_S2N, m_W2E, m_E2W], dim=-1)

        probs = torch.stack([actor_fc2(features_add_comm)
                             for actor_fc2 in self.actor_fc2], dim=0)

        action_mean = torch.tanh(probs)
        action_std = self.log_std.exp()
        action_std = action_std.unsqueeze(
            1).unsqueeze(1).expand_as(action_mean)

        action_mean = torch.gather(action_mean, 0, category.unsqueeze(
            0).unsqueeze(-1).long()).squeeze(0)
        action_std = torch.gather(action_std, 0, category.unsqueeze(
            0).unsqueeze(-1).long()).squeeze(0)

        dist = Normal(action_mean, action_std)

        if deterministic:
            actions = action_mean
        else:
            actions = dist.sample()
        action_log_probs = dist.log_prob(actions)

        values = self.critic(state)
        values = values.unsqueeze(-1).repeat(1, obs.shape[1], 1)

        return check(actions), values, action_log_probs, hidden_states

    def pure_sample(self, obs, m_N2S, m_S2N, m_W2E, m_E2W, hidden_states, deterministic=False):

        category, latent, _ = self.ICP.sample(obs)

        x = self.fc1(latent)
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1)

        features, hidden_states = self.rnn(x, hidden_states)
        features = features.squeeze().reshape(
            obs.shape[:2] + (features.shape[-1],))

        features_add_comm = torch.cat(
            [features, m_N2S, m_S2N, m_W2E, m_E2W], dim=-1)

        probs = torch.stack([actor_fc2(features_add_comm)
                             for actor_fc2 in self.actor_fc2], dim=0)

        action_mean = torch.tanh(probs)
        action_std = self.log_std.exp()
        action_std = action_std.unsqueeze(
            1).unsqueeze(1).expand_as(action_mean)

        action_mean = torch.gather(action_mean, 0, category.unsqueeze(
            0).unsqueeze(-1).long()).squeeze(0)
        action_std = torch.gather(action_std, 0, category.unsqueeze(
            0).unsqueeze(-1).long()).squeeze(0)

        dist = Normal(action_mean, action_std)

        if deterministic:
            actions = action_mean
        else:
            actions = dist.sample()
        action_log_probs = dist.log_prob(actions)

        return check(actions), action_log_probs, hidden_states

    def evaluate(self, obs, state, m_N2S, m_S2N, m_W2E, m_E2W, actions, hidden_states):

        category, latent, _ = self.ICP.sample(obs)

        x = self.fc1(latent)
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1)
        hidden_states = hidden_states.reshape(-1,
                                              hidden_states.shape[-1]).unsqueeze(0)

        features, hidden_states = self.rnn(x, hidden_states)
        features = features.squeeze().reshape(
            obs.shape[:3] + (features.shape[-1],))
        features_add_comm = torch.cat(
            [features, m_N2S, m_S2N, m_W2E, m_E2W], dim=-1)

        probs = torch.stack([actor_fc2(features_add_comm)
                             for actor_fc2 in self.actor_fc2], dim=0)
        probs = torch.gather(
            probs, 0, category.unsqueeze(0).unsqueeze(-1).long())

        action_mean = torch.tanh(probs)
        action_std = self.log_std.exp()

        action_std = action_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
            1, action_mean.shape[1], action_mean.shape[2], action_mean.shape[3], 1)
        action_std = torch.gather(
            action_std, 0, category.unsqueeze(0).unsqueeze(-1).long())

        action_mean = action_mean.squeeze(0)
        action_std = action_std.squeeze(0)

        dist = Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        values = self.critic(state)
        values = values.unsqueeze(-1).repeat(1, 1, obs.shape[2], 1)

        return action_log_probs, dist_entropy, values

    def save_checkpoints(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.state_dict(), f"{self.save_path}/latest0.pt")

    def save_last_checkpoints(self):
        for i in reversed(range(3)):
            load_in = f"{self.save_path}/latest{i}.pt"
            load_out = f"{self.save_path}/latest{i + 1}.pt"

            if os.path.exists(load_in):
                dict = torch.load(load_in)
                torch.save(dict, load_out)

    def load_checkpoints(self):
        dict = torch.load(f"{self.save_path}/latest3.pt")
        self.load_state_dict(dict)
