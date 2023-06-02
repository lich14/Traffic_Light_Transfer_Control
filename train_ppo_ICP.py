import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from generate_road import generate, generate_short
from modules.agent_with_ICP import Agent
from runner.episode_runner import EpisodeRunner
from tensorboardX import SummaryWriter
from torch.optim import Adam

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
    parser.add_argument('--ICP_loss_w', type=float, default=0.001)

    parser.add_argument('--points', type=int, default=2)
    parser.add_argument('--parrals', type=int, default=32)
    parser.add_argument('--GPU', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=32)

    parser.add_argument('--comm_dim', type=int, default=16)
    parser.add_argument('--comm_norm_w', type=float, default=0.001)

    parser.add_argument('--category_num', type=int, default=2)
    parser.add_argument('--reward_para', type=float, default=.0)
    parser.add_argument('--difficult', type=float, default=1.0)
    # default is 0.0, test 0.5 and 1.0

    # 36: top 6 -> transfer
    # 2000
    # point 2, 3, 4 -> 6
    # change reward

    args = parser.parse_args()

    return args


def writereward(csv_path, period_time, total_time, step):
    if os.path.isfile(csv_path):
        with open(csv_path, 'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([step, period_time, total_time])
    else:
        with open(csv_path, 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['step', 'period_time', 'total_time'])
            csv_write.writerow([step, period_time, total_time])


if __name__ == '__main__':
    args = get_args()
    agents = Agent(args, 50, args.points ** 2, args.comm_dim, args.hidden_dim,
                   args.category_num, args.latent_dim, torch.device(args.GPU))

    generate(args.points, args.difficult)
    # 2

    train_error_count = 0
    torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    csv_dir = f'./csv_files/PPO_ICP/points_{args.points}_difficult_{args.difficult}/category_{args.category_num}_reward_para_{args.reward_para}_comm_norm_w_{args.comm_norm_w}_comm_dim_{args.comm_dim}'
    csv_path = f'{csv_dir}/seed_{args.seed}.csv'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    optimizer = Adam(agents.params, lr=args.lr)
    runner = EpisodeRunner(args, agents, args.parrals,
                           args.points, args.GPU, args.difficult)
    writer = SummaryWriter(
        f"runs/ppo_ICP/points_{args.points}_difficult_{args.difficult}/category_{args.category_num}_reward_para_{args.reward_para}_comm_norm_w_{args.comm_norm_w}_comm_dim_{args.comm_dim}_seed_{args.seed}")

    for main_step in range(2000):
        with torch.no_grad():
            if main_step % 20 == 19:
                # episode_return, time_return = runner.eval()
                episode_return, time_return, period_time = runner.eval_full()
                writer.add_scalar(f"episode_return", episode_return, main_step)
                writer.add_scalar(f"time_return", time_return, main_step)
                writer.add_scalar(f"period_time_return",
                                  period_time, main_step)

                print(f'episode return: {episode_return}')
                print(f'time return: {time_return}')
                print('=' * 30)

                writereward(csv_path, period_time, time_return, main_step)

            # sample
            obs, a, old_a_logp, v_preds, rewards, terminate, hidden_states, _, _, _ = runner.run()
            mask = torch.ones(
                (terminate.shape[0] + 1,) + terminate.shape[1:]).to(args.GPU)
            mask[1:] = 1 - terminate

        # train
        obs = obs.to(args.GPU)
        rewards = rewards.to(args.GPU)
        terminate = terminate.to(args.GPU)

        returns = torch.zeros_like(v_preds)
        rewards = rewards.unsqueeze(-1).unsqueeze(-1).expand_as(v_preds)
        terminate = terminate.unsqueeze(-1).unsqueeze(-1).expand_as(v_preds)
        mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                       1, v_preds.shape[-2], 1)
        v_preds = torch.cat(
            [v_preds, torch.zeros_like(v_preds[0]).unsqueeze(0)], dim=0)

        gae = 0
        for step in reversed(range(rewards.shape[0])):
            delta = rewards[step] + args.gamma * \
                v_preds[step + 1] * mask[step + 1] - v_preds[step]
            gae = delta + args.gamma * args.gae_lambda * mask[step + 1] * gae
            returns[step] = gae + v_preds[step]

        adv = returns - v_preds[:-1]
        adv_copy = adv.detach().clone().to('cpu').numpy()
        adv_copy[mask.to('cpu')[:-1] == 0] = np.nan
        mean_adv = np.nanmean(adv_copy)
        std_adv = np.nanstd(adv_copy)
        adv = (adv - mean_adv) / (std_adv + 1e-5)
        adv = adv.squeeze(-1)
        actor_loss_list, critic_loss_list, entropy_loss_list, comm_norm_list, ICP_loss_list, comm_norm_list = [], [], [], [], [], []

        try:
            for _ in range(args.ppo_epoch):

                comm_N2S, comm_S2N, comm_W2E, comm_E2W = [], [], [], []
                h_comm_N2S = torch.zeros(
                    1, obs.shape[0] * obs.shape[1] * args.points, args.comm_dim).to(args.GPU)
                h_comm_S2N = torch.zeros(
                    1, obs.shape[0] * obs.shape[1] * args.points, args.comm_dim).to(args.GPU)
                h_comm_W2E = torch.zeros(
                    1, obs.shape[0] * obs.shape[1] * args.points, args.comm_dim).to(args.GPU)
                h_comm_E2W = torch.zeros(
                    1, obs.shape[0] * obs.shape[1] * args.points, args.comm_dim).to(args.GPU)

                comm_N2S.append(h_comm_N2S)
                comm_S2N.append(h_comm_S2N)
                comm_W2E.append(h_comm_W2E)
                comm_E2W.append(h_comm_E2W)

                for comm_step in range(args.points - 1):
                    inputs_N2S, inputs_S2N, inputs_W2E, inputs_E2W = [], [], [], []
                    for i in range(args.points):
                        inputs_N2S.append(torch.cat([obs[:, :, i + args.points * comm_step, 0:4], obs[:, :, i +
                                          args.points * comm_step, 16:20], obs[:, :, i + args.points * comm_step, 32:36]], dim=-1))
                        inputs_S2N.append(torch.cat([obs[:, :, i + args.points * (args.points - comm_step - 1), 12:16], obs[:, :, i + args.points * (
                            args.points - comm_step - 1), 28:32], obs[:, :, i + args.points * (args.points - comm_step - 1), 44:48]], dim=-1))
                        inputs_W2E.append(torch.cat([obs[:, :, i * args.points + comm_step, 8:12], obs[:, :, i *
                                          args.points + comm_step, 24:28], obs[:, :, i * args.points + comm_step, 40:44]], dim=-1))
                        inputs_E2W.append(torch.cat([obs[:, :, (i + 1) * args.points - 1 - comm_step, 4:8], obs[:, :, (i + 1) *
                                          args.points - 1 - comm_step, 20:24], obs[:, :, (i + 1) * args.points - 1 - comm_step, 36:40]], dim=-1))

                    inputs_N2S = torch.stack(inputs_N2S, dim=2)
                    inputs_S2N = torch.stack(inputs_S2N, dim=2)
                    inputs_W2E = torch.stack(inputs_W2E, dim=2)
                    inputs_E2W = torch.stack(inputs_E2W, dim=2)
                    inputs_comm_shape = inputs_N2S.shape

                    inputs_N2S = inputs_N2S.reshape(-1,
                                                    inputs_N2S.shape[-1]).unsqueeze(1)
                    inputs_S2N = inputs_S2N.reshape(-1,
                                                    inputs_S2N.shape[-1]).unsqueeze(1)
                    inputs_W2E = inputs_W2E.reshape(-1,
                                                    inputs_W2E.shape[-1]).unsqueeze(1)
                    inputs_E2W = inputs_E2W.reshape(-1,
                                                    inputs_E2W.shape[-1]).unsqueeze(1)

                    h_comm_N2S = agents.comm_n2s(
                        inputs_N2S, h_comm_N2S).permute(1, 0, 2)
                    h_comm_S2N = agents.comm_s2n(
                        inputs_S2N, h_comm_S2N).permute(1, 0, 2)
                    h_comm_W2E = agents.comm_w2e(
                        inputs_W2E, h_comm_W2E).permute(1, 0, 2)
                    h_comm_E2W = agents.comm_e2w(
                        inputs_E2W, h_comm_E2W).permute(1, 0, 2)

                    comm_N2S.append(h_comm_N2S)
                    comm_S2N.append(h_comm_S2N)
                    comm_W2E.append(h_comm_W2E)
                    comm_E2W.append(h_comm_E2W)

                comm_N2S = torch.cat(comm_N2S, dim=0)
                comm_S2N = torch.cat(comm_S2N, dim=0)
                comm_W2E = torch.cat(comm_W2E, dim=0)
                comm_E2W = torch.cat(comm_E2W, dim=0)

                comm_N2S = comm_N2S.reshape(
                    (comm_N2S.shape[0],) + inputs_comm_shape[:-1] + (comm_N2S.shape[-1],))
                comm_S2N = comm_S2N.reshape(
                    (comm_S2N.shape[0],) + inputs_comm_shape[:-1] + (comm_S2N.shape[-1],))
                comm_W2E = comm_W2E.reshape(
                    (comm_W2E.shape[0],) + inputs_comm_shape[:-1] + (comm_W2E.shape[-1],))
                comm_E2W = comm_E2W.reshape(
                    (comm_E2W.shape[0],) + inputs_comm_shape[:-1] + (comm_E2W.shape[-1],))

                comm_N2S = comm_N2S.permute(1, 2, 0, 3, 4)
                comm_S2N = comm_S2N.permute(1, 2, 0, 3, 4)
                comm_W2E = comm_W2E.permute(1, 2, 0, 3, 4)
                comm_E2W = comm_E2W.permute(1, 2, 0, 3, 4)

                comm_N2S = torch.cat([comm_N2S[:, :, i]
                                      for i in range(args.points)], dim=2)
                comm_S2N = torch.cat([comm_S2N[:, :, args.points - i - 1]
                                      for i in range(args.points)], dim=2)
                comm_W2E = torch.cat([comm_W2E[:, :, :, i]
                                      for i in range(args.points)], dim=2)
                comm_E2W = torch.cat([comm_E2W[:, :, list(reversed(
                    list(range(args.points)))), i] for i in range(args.points)], dim=2)

                comm_N2S_norm = F.l1_loss(comm_N2S, torch.zeros_like(
                    comm_N2S), reduction='none').sum(dim=-1, keepdim=True)
                comm_S2N_norm = F.l1_loss(comm_S2N, torch.zeros_like(
                    comm_S2N), reduction='none').sum(dim=-1, keepdim=True)
                comm_W2E_norm = F.l1_loss(comm_W2E, torch.zeros_like(
                    comm_W2E), reduction='none').sum(dim=-1, keepdim=True)
                comm_E2W_norm = F.l1_loss(comm_E2W, torch.zeros_like(
                    comm_E2W), reduction='none').sum(dim=-1, keepdim=True)

                comm_N2S_norm = (comm_N2S_norm * mask).sum() / mask.sum()
                comm_S2N_norm = (comm_S2N_norm * mask).sum() / mask.sum()
                comm_W2E_norm = (comm_W2E_norm * mask).sum() / mask.sum()
                comm_E2W_norm = (comm_E2W_norm * mask).sum() / mask.sum()
                comm_norm = comm_N2S_norm + comm_S2N_norm + comm_W2E_norm + comm_E2W_norm

                state = obs.reshape([obs.shape[0], obs.shape[1], -1])

                action_log_probs, entropy, v = agents.evaluate(
                    obs[:-1], state[:-1], comm_N2S[:-1], comm_S2N[:-1], comm_W2E[:-1], comm_E2W[:-1], a, hidden_states.detach())

                value_loss = (v - returns.detach()) ** 2
                value_loss = (
                    value_loss * mask[:-1]).sum() / (mask[:-1].sum() + 1e-4)
                print(f'critic loss: {value_loss}')

                ratio = torch.exp(action_log_probs -
                                  old_a_logp.detach())
                surr1 = ratio * (adv.detach().unsqueeze(-1))
                surr2 = torch.clamp(
                    ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * (adv.detach().unsqueeze(-1))
                action_loss = -torch.min(surr1, surr2)

                action_loss = (
                    action_loss * mask[:-1]).sum() / (mask[:-1].sum() + 1e-4)
                entropy_loss = - \
                    (entropy * mask[:-1]).sum() / (mask[:-1].sum() + 1e-4)
                ICP_loss = agents.ICP.get_loss(obs[:-1], mask[:-1])

                print(f'action_loss: {action_loss}')
                print(f'entropy_loss: {entropy_loss}')
                print(f'ICP_loss: {ICP_loss}')

                optimizer.zero_grad()
                (action_loss + args.entropy_w * entropy_loss +
                 args.ICP_loss_w * ICP_loss + 2 * value_loss + args.comm_norm_w * comm_norm).backward()

                nn.utils.clip_grad_norm_(
                    agents.parameters(), args.max_grad_norm)
                optimizer.step()

                actor_loss_list.append(action_loss.to('cpu').item())
                critic_loss_list.append(value_loss.to('cpu').item())
                entropy_loss_list.append(entropy_loss.to('cpu').item())
                ICP_loss_list.append(ICP_loss.to('cpu').item())

            agents.save_last_checkpoints()
            agents.save_checkpoints()

        except:
            train_error_count += 1
            print(f'train error: {train_error_count} times')
            agents.load_checkpoints()
            try:
                runner.env.close()
            except:
                pass
            runner = EpisodeRunner(
                args, agents, args.parrals, args.points, args.GPU, args.difficult)

        if main_step % 20 == 19:
            writer.add_scalar(f"loss/critic_loss",
                              np.array(critic_loss_list).mean(), main_step)
            writer.add_scalar(f"loss/actor_loss",
                              np.array(actor_loss_list).mean(), main_step)
            writer.add_scalar(f"loss/entropy_loss",
                              np.array(entropy_loss_list).mean(), main_step)
            writer.add_scalar(f"loss/ICP_loss",
                              np.array(ICP_loss_list).mean(), main_step)

            print('=' * 30)
            print(f'current steps: {main_step}')
            print(f'critic loss: {np.array(critic_loss_list).mean()}')
            print(f'actor loss: {np.array(actor_loss_list).mean()}')
            print(f'entropy loss: {np.array(entropy_loss_list).mean()}')
            print(f'ICP loss: {np.array(ICP_loss_list).mean()}')

        if main_step % 100 == 0:

            save_path = f"./models_ppo_ICP/points_{args.points}_difficult_{args.difficult}/category_{args.category_num}_reward_para_{args.reward_para}_comm_norm_w_{args.comm_norm_w}_comm_dim_{args.comm_dim}_seed_{args.seed}/"

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save(agents.state_dict(), f"{save_path}/step_{main_step}.pt")

    print(f'train error: {train_error_count} times')
