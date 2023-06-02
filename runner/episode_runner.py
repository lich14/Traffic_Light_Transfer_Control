import copy
import time
import torch
import numpy as np
from envs.multi_runner import ShareSubprocVecEnv
from modules.util import init
torch.set_num_threads(8)


class ID_points:

    def __init__(self, nums=2):
        self.nums = nums
        self.value = 0

    def add(self):
        self.value = self.value + 1

        check = 0
        while True:
            check += 1
            if self.value < 10 ** (check - 1):
                break

            if ((self.value - (self.value % (10 ** (check - 1)))) / (10 ** (check - 1))) % 10 == self.nums:
                self.value += (10 - self.nums) * (10 ** (check - 1))


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class EpisodeRunner:

    def __init__(self, args, agents, parrals=2, points=3, device='cpu', diff=1.0):
        self.args = args
        self.agents = agents
        self.points = points
        self.parrals = parrals
        self.device = device
        self.diff = diff

        self.env = ShareSubprocVecEnv(
            parrals, points, reward_para=args.reward_para, diff=diff)

    def get_env_info(self):
        return self.env.get_basic_info()

    def eval_full(self, parral=None):
        need_test_episode = 2 ** (self.points * 2)
        id_points = ID_points()
        id_list = []
        returns_list, times_list, period_time_list = [], [], []

        for _ in range(2 ** (2 * self.points)):
            id_str = str(id_points.value)
            if len(id_str) < (2 * self.points):
                for _ in range((2 * self.points) - len(id_str)):
                    id_str = '0' + id_str

            id_list.append(id_str)
            id_points.add()

        if parral is None:
            loop_parral = self.parrals
        else:
            loop_parral = parral

        loop_parral = min(2 ** (2 * self.points), loop_parral)
        loop_num = need_test_episode // loop_parral

        for i in range(loop_num):
            while True:
                self.env.close()
                self.env = ShareSubprocVecEnv(
                    loop_parral, self.points, sumoconfigs=id_list[i * loop_parral: (i + 1) * loop_parral], reward_para=self.args.reward_para, diff=self.diff)

                _, _, _, _, _, _, returns, times, period_times = self.run(
                    True, loop_parral)

                if times is None:
                    pass
                else:
                    returns_list += returns
                    times_list += times
                    period_time_list += period_times

                    print(np.array(times_list).mean())
                    print('-' * 30)
                    break

        return np.array(returns_list).mean(), np.array(times_list).mean(), np.array(period_time_list).mean()

    def run(self, test_mode=False, parrals=None):
        terminated = False
        if parrals == None:
            parrals = self.parrals

        episode_return = [0 for _ in range(parrals)]
        time_return = [0 for _ in range(parrals)]

        obs, succes_reset = self.env.reset()
        if succes_reset:
            pass
        else:
            rereset_time = 0
            while True:
                try:
                    rereset_time += 1
                    self.env.close()
                except Exception as e:
                    print('error 4:', e.__class__.__name__, e, rereset_time)

                self.env = ShareSubprocVecEnv(
                    parrals, self.points, reward_para=self.args.reward_para, diff=self.diff)
                obs, succes_reset = self.env.reset()
                if succes_reset:
                    break

        obs_default = copy.deepcopy(obs[0])
        hidden_states = torch.zeros(
            1, obs.shape[0] * obs.shape[1], self.args.hidden_dim).to(self.args.GPU)

        obs_list, action_list, action_logp_list, value_pred_list, reward_list, \
            terminate_list, hidden_states_list = [], [], [], [], [], [], []
        obs_list.append(obs)
        old_done_int = [0 for _ in range(parrals)]

        while not terminated:
            hidden_states_list.append(hidden_states)

            obs = obs.to(self.device)
            comm_N2S, comm_S2N, comm_W2E, comm_E2W = [], [], [], []

            h_comm_N2S = torch.zeros(
                1, obs.shape[0] * self.points, self.args.comm_dim).to(self.device)
            h_comm_S2N = torch.zeros(
                1, obs.shape[0] * self.points, self.args.comm_dim).to(self.device)
            h_comm_W2E = torch.zeros(
                1, obs.shape[0] * self.points, self.args.comm_dim).to(self.device)
            h_comm_E2W = torch.zeros(
                1, obs.shape[0] * self.points, self.args.comm_dim).to(self.device)

            comm_N2S.append(h_comm_N2S)
            comm_S2N.append(h_comm_S2N)
            comm_W2E.append(h_comm_W2E)
            comm_E2W.append(h_comm_E2W)

            for comm_step in range(self.points - 1):
                inputs_N2S, inputs_S2N, inputs_W2E, inputs_E2W = [], [], [], []
                for i in range(self.points):
                    inputs_N2S.append(torch.cat([obs[:, i + self.points * comm_step, 0:4], obs[:, i +
                                      self.points * comm_step, 16:20], obs[:, i + self.points * comm_step, 32:36]], dim=-1))
                    inputs_S2N.append(torch.cat([obs[:, i + self.points * (self.points - comm_step - 1), 12:16], obs[:, i + self.points * (
                        self.points - comm_step - 1), 28:32], obs[:, i + self.points * (self.points - comm_step - 1), 44:48]], dim=-1))
                    inputs_W2E.append(torch.cat([obs[:, i * self.points + comm_step, 8:12], obs[:, i *
                                      self.points + comm_step, 24:28], obs[:, i * self.points + comm_step, 40:44]], dim=-1))
                    inputs_E2W.append(torch.cat([obs[:, (i + 1) * self.points - 1 - comm_step, 4:8], obs[:, (i + 1) *
                                      self.points - 1 - comm_step, 20:24], obs[:, (i + 1) * self.points - 1 - comm_step, 36:40]], dim=-1))

                inputs_N2S = torch.stack(inputs_N2S, dim=1)
                inputs_S2N = torch.stack(inputs_S2N, dim=1)
                inputs_W2E = torch.stack(inputs_W2E, dim=1)
                inputs_E2W = torch.stack(inputs_E2W, dim=1)

                inputs_N2S = inputs_N2S.reshape(-1,
                                                inputs_N2S.shape[-1]).unsqueeze(1)
                inputs_S2N = inputs_S2N.reshape(-1,
                                                inputs_S2N.shape[-1]).unsqueeze(1)
                inputs_W2E = inputs_W2E.reshape(-1,
                                                inputs_W2E.shape[-1]).unsqueeze(1)
                inputs_E2W = inputs_E2W.reshape(-1,
                                                inputs_E2W.shape[-1]).unsqueeze(1)

                h_comm_N2S = self.agents.comm_n2s(
                    inputs_N2S, h_comm_N2S).permute(1, 0, 2)
                h_comm_S2N = self.agents.comm_s2n(
                    inputs_S2N, h_comm_S2N).permute(1, 0, 2)
                h_comm_W2E = self.agents.comm_w2e(
                    inputs_W2E, h_comm_W2E).permute(1, 0, 2)
                h_comm_E2W = self.agents.comm_e2w(
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
                self.points, -1, self.points, comm_N2S.shape[-1]).permute(1, 0, 2, 3)
            comm_S2N = comm_S2N.reshape(
                self.points, -1, self.points, comm_N2S.shape[-1]).permute(1, 0, 2, 3)
            comm_W2E = comm_W2E.reshape(
                self.points, -1, self.points, comm_N2S.shape[-1]).permute(1, 0, 2, 3)
            comm_E2W = comm_E2W.reshape(
                self.points, -1, self.points, comm_N2S.shape[-1]).permute(1, 0, 2, 3)

            comm_N2S = torch.cat([comm_N2S[:, i]
                                 for i in range(self.points)], dim=1)
            comm_S2N = torch.cat([comm_S2N[:, self.points - i - 1]
                                 for i in range(self.points)], dim=1)
            comm_W2E = torch.cat([comm_W2E[:, :, i]
                                 for i in range(self.points)], dim=1)
            comm_E2W = torch.cat([comm_E2W[:, list(reversed(list(range(self.points)))), i]
                                 for i in range(self.points)], dim=1)

            '''if torch.isnan(obs.mean()):
                print('hh, find it, nan is obs')
            if torch.isnan(comm_N2S.mean()):
                print('hh, find it, nan is comm_N2S')
            if torch.isnan(comm_S2N.mean()):
                print('hh, find it, nan is comm_S2N')
            if torch.isnan(comm_W2E.mean()):
                print('hh, find it, nan is comm_W2E')
            if torch.isnan(comm_E2W.mean()):
                print('hh, find it, nan is comm_E2W')'''

            state = obs.reshape([obs.shape[0], -1])

            if test_mode == False:
                actions, value_pred, action_log_probs, hidden_states = self.agents.sample(
                    obs, state, comm_N2S, comm_S2N, comm_W2E, comm_E2W, hidden_states, test_mode)
            else:
                actions, action_log_probs, hidden_states = self.agents.pure_sample(
                    obs, comm_N2S, comm_S2N, comm_W2E, comm_E2W, hidden_states, test_mode)

            obs_, reward, ifdone, cur_time, succes_step, period_time = self.env.step(
                actions)

            if succes_step:
                pass
            else:
                if test_mode == False:
                    return torch.stack(obs_list, dim=0), torch.stack(action_list, dim=0), \
                        torch.stack(action_logp_list, dim=0), torch.stack(value_pred_list, dim=0), \
                        torch.tensor(reward_list), torch.tensor(terminate_list), \
                        torch.stack(hidden_states_list,
                                    dim=0), None, None, None
                else:
                    return torch.stack(obs_list, dim=0), torch.stack(action_list, dim=0), \
                        torch.stack(action_logp_list, dim=0), torch.tensor(reward_list), \
                        torch.tensor(terminate_list), torch.stack(
                            hidden_states_list, dim=0), None, None, None

            action_list.append(check(actions))
            action_logp_list.append(action_log_probs)

            if test_mode == False:
                value_pred_list.append(value_pred)

            done_int = [int(item) for item in ifdone]
            obs = torch.stack(
                [item if item is not None else obs_default for item in obs_], dim=0)
            terminated = all(ifdone)

            for i in range(parrals):
                episode_return[i] += reward[i] * (1 - old_done_int[i])
                time_return[i] = cur_time[i] * done_int[i]
            old_done_int = done_int

            obs_list.append(obs)
            reward_list.append(reward)
            terminate_list.append([int(item) for item in ifdone])

        self.env.close()
        self.env = ShareSubprocVecEnv(
            self.parrals, self.points, reward_para=self.args.reward_para, diff=self.diff)

        if test_mode == False:
            return torch.stack(obs_list, dim=0), torch.stack(action_list, dim=0), torch.stack(action_logp_list, dim=0),\
                torch.stack(value_pred_list, dim=0), torch.tensor(reward_list), torch.tensor(terminate_list), \
                torch.stack(hidden_states_list,
                            dim=0), episode_return, time_return, period_time
        else:
            return torch.stack(obs_list, dim=0), torch.stack(action_list, dim=0), torch.stack(action_logp_list, dim=0),\
                torch.tensor(reward_list), torch.tensor(terminate_list), torch.stack(
                    hidden_states_list, dim=0), episode_return, time_return, period_time
