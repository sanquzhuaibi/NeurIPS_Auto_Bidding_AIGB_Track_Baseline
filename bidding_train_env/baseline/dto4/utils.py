import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import pickle
import random



def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward, penalty


class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path, max_ep_len=24, scale=2000, K=20):
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        super(EpisodeReplayBuffer, self).__init__()
        self.max_ep_len = max_ep_len
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        training_data = pd.read_csv(data_path)
        # data_path_temp = './bidding_train_env/data/trajectory/trajectory_data_temp.csv'
        # training_data[:10000].to_csv(data_path_temp)

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
        training_data["state_shift"] = training_data["state"].shift(1)
        training_data['state_shift'][training_data['timeStepIndex'] == 0] = None
        fill_value = (0,) *len(training_data["state"][0])
        training_data['state_shift'] = training_data['state_shift'].apply(lambda x: fill_value if x is None else x)
        # training_data["state_shift"]
        # training_data["state_shift"] = training_data['state_shift'][training_data['timeStepIndex'] == 0].apply(lambda x:(0,) *len(training_data["state"][0]))
        training_data['state_new'] = training_data['state'] + training_data['state_shift']
        # training_data["state_new"] = training_data["state"] - training_data["state"].shift(1, fill_value=0)
        training_data["next_state_new"] = training_data["next_state"]
        training_data.to_csv(data_path)
        self.trajectories = training_data

        self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones, self.scores, self.penalties = [], [], [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        score = []
        penalty = []

        for index, row in self.trajectories.iterrows():
            state_concat = row['state_new'] + ( row['advertiserCategoryIndex'], row['budget'], row['CPAConstraint'])
            state.append(state_concat)
            # state_concat = np.concatenate((row["state_new_arr"] , row["state_new_arr_shift"]), -1)
            # state.append(state_concat)
            reward.append(row['reward_continuous'])
            action.append(row["action"])
            dones.append(row["done"])

            score.append(row["reward_continuous"])
            cost = (1-row['state_new'][1]) * row['budget']

            if row["done"]:
                if len(state) != 1:
                    state_arr = np.array(state)
                    state_dim_temp = state_dim - 3
                    diff = state_arr[:, :state_dim_temp//2] - state_arr[:, state_dim_temp//2:state_dim_temp]
                    state_arr[:, state_dim_temp // 2:state_dim_temp] = diff
                    self.states.append(state_arr)
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                    score_temp, penalty_temp = getScore_nips(sum(reward), cost / sum(reward), row['CPAConstraint'])
                    self.scores.append(score_temp)
                    self.penalties.append(penalty_temp)

                state = []
                reward = []
                action = []
                dones = []
                score = []
        self.traj_lens, self.returns, self.scores, self.penalties= np.array(self.traj_lens), np.array(self.returns), np.array(self.scores), np.array(self.penalties)
        print(f'nums_penalties: {sum(self.penalties < 1)}')
        print(f'nums_penalties < 0.9: {sum(self.penalties < 0.9)}')
        print(f'all_trajectory: {len(self.penalties)}')
        # self.states = np.array(self.states)
        # self.actions = np.array(self.actions)
        # self.dones = np.array(self.dones)
        # self.rewards = np.array(self.rewards)


        nums_length = len(self.scores)
        threshold = int(0.5 * nums_length)
        # sorted_inds_scores = np.argsort(-self.scores) [:threshold] # from highest to lower index
        sorted_inds_scores = np.argsort(-self.penalties) [:threshold] # from highest to lower index
        print(f'nums_penalties_modification < 1: {sum(self.penalties[sorted_inds_scores] < 1)}')

        # self.states = self.states[sorted_inds_scores]
        # self.actions = self.actions[sorted_inds_scores]
        # self.rewards = self.rewards[sorted_inds_scores]
        # self.dones = self.dones[sorted_inds_scores]
        # self.scores = self.scores[sorted_inds_scores]

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-6

        self.trajectories = []

        for i in range(len(self.states)):
            # if i not in sorted_inds_scores:
            #     continue
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                 "dones": self.dones[i]})

        self.K = K
        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens[sorted_inds_scores])
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        # sorted_inds = np.argsort(self.returns)  # lowest to highest
        sorted_inds = sorted_inds_scores[::-1]#from lower to higher
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        # ind = len(self.trajectories) - 2
        ind = len(sorted_inds_scores) - 2
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
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = r / self.scale
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
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



class EpisodeReplayBuffer_new(Dataset):
    def __init__(self, state_dim, act_dim, data_path, max_ep_len=24, scale=2000, K=20):
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        super(EpisodeReplayBuffer, self).__init__()
        self.max_ep_len = max_ep_len
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        training_data = pd.read_csv(data_path)
        # data_path_temp = './bidding_train_env/data/trajectory/trajectory_data_temp.csv'
        # training_data[:10000].to_csv(data_path_temp)
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
        # training_data["state_shift"] = training_data["state"].shift(1)
        # training_data['state_shift'][training_data['timeStepIndex'] == 0] = None
        # fill_value = (0,) *len(training_data["state"][0])
        # training_data['state_shift'] = training_data['state_shift'].apply(lambda x: fill_value if x is None else x)
        # # training_data["state_shift"]
        # # training_data["state_shift"] = training_data['state_shift'][training_data['timeStepIndex'] == 0].apply(lambda x:(0,) *len(training_data["state"][0]))
        # training_data['state_new'] = training_data['state'] + training_data['state_shift']
        # # training_data["state_new"] = training_data["state"] - training_data["state"].shift(1, fill_value=0)
        # training_data["next_state_new"] = training_data["next_state"]
        # training_data.to_csv(data_path)
        self.trajectories = training_data

        self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones, self.scores, self.penalties = [], [], [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        score = []
        penalty = []

        for index, row in self.trajectories.iterrows():
            state_concat = row['state_new'] + ( row['advertiserCategoryIndex'], row['budget'], row['CPAConstraint'])
            state.append(state_concat)
            # state_concat = np.concatenate((row["state_new_arr"] , row["state_new_arr_shift"]), -1)
            # state.append(state_concat)
            reward.append(row['reward_continuous'])
            action.append(row["action"])
            dones.append(row["done"])

            score.append(row["reward_continuous"])
            cost = (1-row['state_new'][1]) * row['budget']

            if row["done"]:
                if len(state) != 1:
                    state_arr = np.array(state)
                    state_dim_temp = state_dim - 3
                    diff = state_arr[:, :state_dim_temp//2] - state_arr[:, state_dim_temp//2:state_dim_temp]
                    state_arr[:, state_dim_temp // 2:state_dim_temp] = diff
                    self.states.append(state_arr)
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                    score_temp, penalty_temp = getScore_nips(sum(reward), cost / sum(reward), row['CPAConstraint'])
                    self.scores.append(score_temp)
                    self.penalties.append(penalty_temp)

                state = []
                reward = []
                action = []
                dones = []
                score = []
        self.traj_lens, self.returns, self.scores, self.penalties= np.array(self.traj_lens), np.array(self.returns), np.array(self.scores), np.array(self.penalties)
        print(f'nums_penalties: {sum(self.penalties < 1)}')
        print(f'nums_penalties < 0.9: {sum(self.penalties < 0.9)}')
        print(f'all_trajectory: {len(self.penalties)}')
        # self.states = np.array(self.states)
        # self.actions = np.array(self.actions)
        # self.dones = np.array(self.dones)
        # self.rewards = np.array(self.rewards)


        nums_length = len(self.scores)
        threshold = int(0.5 * nums_length)
        # sorted_inds_scores = np.argsort(-self.scores) [:threshold] # from highest to lower index
        sorted_inds_scores = np.argsort(-self.penalties) [:threshold] # from highest to lower index
        print(f'nums_penalties_modification < 1: {sum(self.penalties[sorted_inds_scores] < 1)}')

        # self.states = self.states[sorted_inds_scores]
        # self.actions = self.actions[sorted_inds_scores]
        # self.rewards = self.rewards[sorted_inds_scores]
        # self.dones = self.dones[sorted_inds_scores]
        # self.scores = self.scores[sorted_inds_scores]

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-6

        self.trajectories = []

        for i in range(len(self.states)):
            # if i not in sorted_inds_scores:
            #     continue
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                 "dones": self.dones[i]})

        self.K = K
        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens[sorted_inds_scores])
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        # sorted_inds = np.argsort(self.returns)  # lowest to highest
        sorted_inds = sorted_inds_scores[::-1]#from lower to higher
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        # ind = len(self.trajectories) - 2
        ind = len(sorted_inds_scores) - 2
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
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = r / self.scale
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
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


