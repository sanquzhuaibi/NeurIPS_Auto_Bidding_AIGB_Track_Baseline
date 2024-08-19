import numpy as np
import logging

import torch.cuda

from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dtcql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.dtcql.cql import CQL
from bidding_train_env.baseline.dtcql.iql import IQL

import sys
import pandas as pd
import ast
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
import random
import os
import torch
from bidding_train_env.baseline.dtcql.dt import DecisionTransformer


os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# 获取当前的日期和时间
current_datetime = datetime.now()
# 格式化日期和时间为字符串
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

writer = SummaryWriter(f"results/dt_{formatted_datetime}")


os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16


def train_cql_model():
    """
    Train the cql model.
    """
    train_data_path = "./bidding_train_env/data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    is_normalize = True

    if is_normalize:
        normalize_dic = normalize_state_QDT(training_data, STATE_DIM, normalize_indices=list(np.arange(16)))
        # select use continuous reward
        training_data['reward'] = normalize_reward(training_data, "reward_continuous")
        # select use sparse reward
        # training_data['reward'] = normalize_reward(training_data, "reward")
        save_normalize_dict(normalize_dic, "saved_model/QDTtest")

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))


    #FOR training CQL
    # Train model
    # model = CQL(STATE_DIM,  1, device=device)
    model_pre = IQL(STATE_DIM,  1)

    train_model_steps_iql(model_pre, replay_buffer)
    # Save model
    # model_pre.save_jit("saved_model/CQLtest")


    #Data processing
    # train_data_path = "./bidding_train_env/data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    # training_data = pd.read_csv(train_data_path)
    #
    # def safe_literal_eval(val):
    #     if pd.isna(val):
    #         return val
    #     try:
    #         return ast.literal_eval(val)
    #     except (ValueError, SyntaxError):
    #         print(ValueError)
    #         return val
    #
    # training_data["state"] = training_data["state"].apply(safe_literal_eval)
    # training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    # is_normalize = True
    #
    # if is_normalize:
    #     normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=[13, 14, 15])
    #     # select use continuous reward
    #     training_data['reward'] = normalize_reward(training_data, "reward_continuous")
    #     # select use sparse reward
    #     # training_data['reward'] = normalize_reward(training_data, "reward")

    # Build replay buffer
    replay_buffer =  EpisodeReplayBuffer_QDT(16, 1, training_data, model=model_pre,is_normalize=is_normalize)
    step_num = 1
    batch_size = 64
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)#根据这些权重来调整每个样本在训练过程中被抽样的概率，从而更加有效地训练模型。
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std, state_norm_dict=normalize_dic)
    model.to(device)

    model.train()
    # test_state = np.ones(16, dtype=np.float32)
    # logger.info(f"Test action: {model.take_actions(test_state)}")
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size
    with open(f'results/dt_{formatted_datetime}/model_hyperparameters.txt', 'w') as f:
        for key, value in model.hyperparameters.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask in dataloader:
        train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        if i % 1000 == 0:
            logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")

        writer.add_scalar('Action loss', np.mean(train_loss), i)

        i += 1

        model.scheduler.step()

    model.save_net("saved_model/QDTtest")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")

    # Test trained model
    # test_trained_model(model, replay_buffer)


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))



def train_model_steps(model, replay_buffer, step_num=1, batch_size=128):#batch_size = 128
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)
        a_loss, _, q1_loss, q2_loss = model.step(states, actions, rewards, next_states, terminals)
        logger.info(f'Step: {i} Q1_loss: {q1_loss} Q2_loss: {q2_loss} A_loss: {a_loss}')


def train_model_steps_iql(model, replay_buffer, step_num=1, batch_size=128):#batch_size = 128
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)
        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)
        if i %1000 == 0:
            logger.info(f'Step: {i} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred action:", tem)


def run_cql():
    print(sys.path)
    """
    Run cql model training and evaluation.
    """
    train_cql_model()

class EpisodeReplayBuffer_QDT(Dataset):
    def __init__(self, state_dim, act_dim, data_path, max_ep_len=24, scale=2000, K=20):
        self.device = "cpu"
        super(EpisodeReplayBuffer_QDT, self).__init__()
        self.max_ep_len = max_ep_len
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        training_data = pd.read_csv(data_path)

        def safe_literal_eval(val):
            if pd.isna(val):
                return val
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                print(ValueError)
                return val

        training_data["state"] = training_data["state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["state"].shift(-1)
        training_data.at[training_data.index[-1], 'next_state'] = training_data.at[0, 'state']
        self.trajectories = training_data

        self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones, self.next_states = [], [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        next_state = []
        for index, row in self.trajectories.iterrows():
            state.append(row["state"])
            reward.append(row['reward_continuous'])
            action.append(row["action"])
            dones.append(row["done"])
            if row["done"] == 0:
                next_state.append(row["next_state"])
            else:
                next_state.append((0,)*len(row["state"]))
            if row["done"]:
                if len(state) != 1:
                    self.states.append(np.array(state))
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                    self.next_states.append(np.array(next_state))
                state = []
                reward = []
                action = []
                dones = []
                next_state = []
        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-6

        self.trajectories = []
        for i in range(len(self.states)):
            # self.trajectories.append(
            #     {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
            #      "dones": self.dones[i]})
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                 "dones": self.dones[i], "next_observations": self.next_states[i]})

        self.K = K
        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest（从return中最小的值的index排序起来）
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]

        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])

    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, traj['rewards'].shape[0] - 1)

        s = traj['observations'][start_t: start_t + self.K]
        s_prime = traj['next_observations'][start_t: start_t + self.K]

        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        s_prime = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s_prime], axis=0)
        s_prime = (s_prime - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = r / self.scale
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        s_prime = torch.from_numpy(s_prime).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, a, r, d, rtg, timesteps, mask

    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum





def normalize_state_QDT(training_data, state_dim, normalize_indices):
    """
    Normalize features for reinforcement learning.
    Args:
        training_data: A DataFrame containing the training data.
        state_dim: The total dimension of the features.
        normalize_indices: A list of indices of the features to be normalized.

    Returns:
        A dictionary containing the normalization statistics.
    """
    state_columns = [f'state{i}' for i in range(state_dim)]
    next_state_columns = [f'next_state{i}' for i in range(state_dim)]

    for i, (state_col, next_state_col) in enumerate(zip(state_columns, next_state_columns)):
        training_data[state_col] = training_data['state'].apply(
            lambda x: x[i] if x is not None and not np.isnan(x).any() else 0.0)
        training_data[next_state_col] = training_data['next_state'].apply(
            lambda x: x[i] if x is not None and not np.isnan(x).any() else 0.0)

    stats = {
        i: {
            'min': training_data[state_columns[i]].min(),
            'max': training_data[state_columns[i]].max(),
            'mean': training_data[state_columns[i]].mean(),
            'std': training_data[state_columns[i]].std()
        }
        for i in normalize_indices
    }

    for state_col, next_state_col in zip(state_columns, next_state_columns):
        if int(state_col.replace('state', '')) in normalize_indices:
            min_val = stats[int(state_col.replace('state', ''))]['min']
            max_val = stats[int(state_col.replace('state', ''))]['max']
            training_data[f'normalize_{state_col}'] = (
                                                              training_data[state_col] - min_val) / (
                                                              max_val - min_val + 1e-10)
            training_data[f'normalize_{next_state_col}'] = (
                                                                   training_data[next_state_col] - min_val) / (
                                                                   max_val - min_val + 1e-10)
            # 0.01 error too large?
        else:
            training_data[f'normalize_{state_col}'] = training_data[state_col]
            training_data[f'normalize_{next_state_col}'] = training_data[next_state_col]

    training_data['normalize_state'] = training_data.apply(
        lambda row: tuple(row[f'normalize_{state_col}'] for state_col in state_columns), axis=1)
    training_data['normalize_nextstate'] = training_data.apply(
        lambda row: tuple(row[f'normalize_{next_state_col}'] for next_state_col in next_state_columns), axis=1)

    return stats

if __name__ == '__main__':
    run_cql()
