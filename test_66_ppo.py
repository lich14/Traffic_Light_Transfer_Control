import os
from time import time_ns
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from generate_road import generate, generate_short
from torch.optim import Adam
from modules.agent import Agent
from tensorboardX import SummaryWriter
from runner.episode_runner import EpisodeRunner
torch.set_num_threads(8)


def get_args():
    parser = argparse.ArgumentParser(
        description='Train a PPO agent for traffic light')

    parser.add_argument('--ppo_epoch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=1.)
    parser.add_argument('--entropy_w', type=float, default=.03)
    parser.add_argument('--comm_dim', type=int, default=16)

    parser.add_argument('--points', type=int, default=2)
    parser.add_argument('--parrals', type=int, default=32)
    parser.add_argument('--GPU', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--difficult', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--comm_norm_w', type=float, default=0.001)

    parser.add_argument('--reward_para', type=float, default=.0)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    agents = Agent(args, 50, args.points ** 2, args.comm_dim,
                   args.hidden_dim, torch.device(args.GPU))
    generate(6, args.difficult)

    torch.manual_seed(args.seed)

    agents.load_state_dict(torch.load(
        'models_ppo/points_2/reward_para_0.0_seed_0/step_1500.pt'))
    runner = EpisodeRunner(args, agents, 64, 6, args.GPU, args.difficult)
    episode_return, time, period_time = runner.eval_full()
    print(episode_return, time, period_time)
