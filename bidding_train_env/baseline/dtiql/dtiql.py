

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


"""0.3911 step_num=100000*2, batch_size=256"""
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import numpy as np
import torch
from copy import deepcopy
import os
from torch.distributions import Normal


class Q(nn.Module):
    '''
    IQL-Q net
    '''

    def __init__(self, dim_observation, dim_action):
        super(Q, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.obs_FC = nn.Linear(self.dim_observation, 64)
        self.action_FC = nn.Linear(dim_action, 64)
        self.FC1 = nn.Linear(128, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs_embedding = self.obs_FC(obs)
        action_embedding = self.action_FC(acts)
        embedding = torch.cat([obs_embedding, action_embedding], dim=-1)
        q = self.FC3(F.relu(self.FC2(F.relu(self.FC1(embedding)))))
        return q


class V(nn.Module):
    '''
        IQL-V net
        '''

    def __init__(self, dim_observation):
        super(V, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        return self.FC4(result)


class Actor(nn.Module):
    '''
    IQL-actor net
    '''

    def __init__(self, dim_observation, dim_action, log_std_min=-10, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC_mu = nn.Linear(64, dim_action)
        self.FC_std = nn.Linear(64, dim_action)

    def forward(self, obs):
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mu = self.FC_mu(x)
        log_std = self.FC_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, obs, epsilon=1e-6):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action, dist

    def get_action(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action.detach().cpu()

    def get_det_action(self, obs):
        mu, _ = self.forward(obs)
        return mu.detach().cpu()


class DecisionTransformerIQL(nn.Module):
    '''
    IQL model
    '''

    def __init__(self, dim_obs=3, dim_actions=1, gamma=0.99, tau=0.01, V_lr=1e-4, critic_lr=1e-4, actor_lr=5e-4,
                 network_random_seed=1, expectile=0.7, temperature=3.0, state_mean=0, state_std=1):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.V_lr = V_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.network_random_seed = network_random_seed
        self.expectile = expectile
        self.temperature = temperature
        torch.random.manual_seed(self.network_random_seed)
        self.value_net = V(self.num_of_states)
        self.critic1 = Q(self.num_of_states, self.num_of_actions)
        self.critic2 = Q(self.num_of_states, self.num_of_actions)
        self.critic1_target = Q(self.num_of_states, self.num_of_actions)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Q(self.num_of_states, self.num_of_actions)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # self.actors = Actor(self.num_of_states, self.num_of_actions)
        self.actors = DecisionTransformer(state_dim=dim_obs, act_dim=dim_actions, state_mean=state_mean,
                                state_std=state_std)
        self.GAMMA = gamma
        self.tau = tau
        self.value_optimizer = AdamW(self.value_net.parameters(), lr=self.V_lr)
        self.critic1_optimizer = AdamW(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = AdamW(self.critic2.parameters(), lr=self.critic_lr)
        self.actor_optimizer = AdamW(self.actors.parameters(), lr=self.actor_lr)

        self.warmup_steps = 10000
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic1_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic1_optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.critic2_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic2_optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))
        self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.hyperparameters = {
            "IQL_dim_obs":dim_obs,
            "IQL_dim_actions":dim_actions,
            "IQL_gamma":gamma,
            "IQL_tau":tau,
            "IQL_V_lr":V_lr,
            "IQL_critic_lr":critic_lr,
            "IQL_actor_lr":actor_lr,
            "network_random_seed":network_random_seed,
            "expectile":expectile,
            "temperature":temperature,
        }
        self.hyperparameters.update(self.actors.hyperparameters)



        self.deterministic_action = True
        if self.device == 'cpu':
            self.use_cuda = False
        else:
            self.use_cuda = True
        # if self.use_cuda:
        #     self.critic1.cuda()
        #     self.critic2.cuda()
        #     self.critic1_target.cuda()
        #     self.critic2_target.cuda()
        #     self.value_net.cuda()
        #     self.actors.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

    def step(self, states, actions, rewards, next_states, dones,  rtg, timesteps, attention_mask):
        '''
        train model
        '''
        states = states.to(dtype = torch.float32, device = self.device)
        actions = actions.to(dtype = torch.float32, device = self.device)
        rewards = rewards.to(dtype = torch.float32, device = self.device)
        next_states = next_states.to(dtype = torch.float32, device = self.device)
        dones = dones.to(device = self.device)
        rtg = rtg.to(dtype = torch.float32, device = self.device)
        timesteps = timesteps.to(device = self.device)
        attention_mask = attention_mask.to(device = self.device)


        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions, attention_mask)
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = self.calc_policy_loss(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states, attention_mask)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)

        return critic1_loss.cpu().data.numpy(), value_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    def take_actions(self, states):
        '''
        take action
        '''
        states = torch.Tensor(states).type(self.FloatTensor)
        if self.deterministic_action:
            actions = self.actors.get_det_action(states)
        else:
            actions = self.actors.get_action(states)
        actions = torch.clamp(actions, 0)
        actions = actions.cpu().data.numpy()
        return actions

    def forward(self, states):

        actions = self.actors.get_det_action(states)
        actions = torch.clamp(actions, min=0)
        return actions

    def calc_policy_loss(self,states, actions, rewards, dones, rtg, timesteps, attention_mask=None):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)

            q1 = q1.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]
            q2 = q2.reshape(-1, q2.shape[-1])[attention_mask.reshape(-1) > 0]
            v = v.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]

            min_Q = torch.min(q1, q2)

        exp_a = torch.exp(min_Q - v) * self.temperature
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self.device))

        _, dist = self.actors.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        log_probs = dist.log_prob(actions)
        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        log_probs = log_probs.reshape(-1, actions.shape[-1])[attention_mask.reshape(-1) > 0]
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    def calc_value_loss(self, states, actions, attention_mask=None):
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)

            q1 = q1.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]
            q2 = q2.reshape(-1, q2.shape[-1])[attention_mask.reshape(-1) > 0]

            min_Q = torch.min(q1, q2)



        value = self.value_net(states)
        value = value.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]

        value_loss = self.l2_loss(min_Q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states, attention_mask=None):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.GAMMA * (1 - dones.unsqueeze(-1)) * next_v)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        q1  = q1.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]
        q2 = q2.reshape(-1, q2.shape[-1])[attention_mask.reshape(-1) > 0]
        q_target = q_target.reshape(-1, q2.shape[-1])[attention_mask.reshape(-1) > 0]

        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        return critic1_loss, critic2_loss

    def update_target(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1. - self.tau) * target_param.data + self.tau * local_param.data)

    # def save_net(self, save_path):
    #     '''
    #     save model
    #     '''
    #     if not os.path.isdir(save_path):
    #         os.makedirs(save_path)
    #     torch.save(self.critic1, save_path + "/critic1" + ".pkl")
    #     torch.save(self.critic2, save_path + "/critic2" + ".pkl")
    #     torch.save(self.value_net, save_path + "/value_net" + ".pkl")
    #     torch.save(self.actors, save_path + "/actor" + ".pkl")
    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "dt.pt")
        torch.save(self.state_dict(), file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/iql_model.pth')

    # def load_net(self, load_path="saved_model/fixed_initial_budget", device='cuda:0'):
    #     '''
    #     load model
    #     '''
    #     if os.path.isfile(load_path + "/critic.pt"):
    #         self.critic1.load_state_dict(torch.load(load_path + "/critic1.pt", map_location='cpu'))
    #         self.critic2.load_state_dict(torch.load(load_path + "/critic2.pt", map_location='cpu'))
    #         self.actors.load_state_dict(torch.load(load_path + "/actor.pt", map_location='cpu'))
    #     else:
    #         self.critic1 = torch.load(load_path + "/critic1.pkl", map_location='cpu')
    #         self.critic2 = torch.load(load_path + "/critic2.pkl", map_location='cpu')
    #         self.actors = torch.load(load_path + "/actor.pkl", map_location='cpu')
    #     self.value_net = torch.load(load_path + "/value_net.pkl", map_location='cpu')
    #     print("model stored path " + next(self.critic1.parameters()).device.type)
    #     self.critic1_target = deepcopy(self.critic1)
    #     self.critic2_target = deepcopy(self.critic2)
    #     self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
    #     self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
    #     self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
    #     self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)
    #
    #     # cuda usage
    #     self.use_cuda = torch.cuda.is_available()
    #     if self.use_cuda:
    #         self.critic1.cuda()
    #         self.critic2.cuda()
    #         self.value_net.cuda()
    #         self.actors.cuda()
    #         self.critic1_target.cuda()
    #         self.critic2_target.cuda()
    #     print("model stored path " + next(self.critic1.parameters()).device.type)
    def load_net(self, load_path="saved_model/DTIQLtest", device='cpu'):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device))
        print(f"Model loaded from {self.device}.")


    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])

        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = mask.view(B, -1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, action_tanh=False, K=30, max_ep_len=96, scale=2000,
                 target_return=4):
        super(DecisionTransformer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.length_times = 3
        self.hidden_size = 64
        self.state_mean = state_mean
        self.state_std = state_std
        # assert self.hidden_size == config['n_embd']
        self.max_length = K
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return

        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = 5e-4


        self.block_config = {
            "n_ctx": 1024,
            "n_embd": 64,
            "n_layer": 3,
            "n_head": 1,
            "n_inner": 512,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1
        }
        block_config = self.block_config
        self.hyperparameters = {
            "n_ctx": self.block_config['n_ctx'],
            "n_embd": self.block_config['n_embd'],
            "n_layer": self.block_config['n_layer'],
            "n_head": self.block_config['n_head'],
            "n_inner": self.block_config['n_inner'],
            "activation_function": self.block_config['activation_function'],
            "n_position": self.block_config['n_position'],
            "resid_pdrop": self.block_config['resid_pdrop'],
            "attn_pdrop": self.block_config['attn_pdrop'],
            "length_times": self.length_times,
            "hidden_size": self.hidden_size,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "max_length": self.max_length,
            "K": K,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "scale": scale,
            "target_return": target_return,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,

        }

        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_return = torch.nn.Linear(self.hidden_size, 1)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.init_eval()

        self.FC_mu = nn.Linear(64, act_dim)
        self.FC_std = nn.Linear(64, act_dim)
        self.log_std_min=-10
        self.log_std_max=2

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)



        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])


        mu = self.FC_mu(x[:, 1])
        log_std = self.FC_std(x[:, 1])
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)



        return state_preds, action_preds, return_preds, None, mu, log_std

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1), device=rewards.device),
                 rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, reward_preds, mu, log_std= self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return mu[0, -1]

    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask):
        rewards_target, action_target, rtg_target = torch.clone(rewards), torch.clone(actions), torch.clone(rtg)

        state_preds, action_preds, return_preds, reward_preds, mu, log_std = self.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )


        act_dim = action_preds.shape[2]
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()




        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        #
        # loss = torch.mean((action_preds - action_target) ** 2)
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        # self.optimizer.step()

        return action_preds, dist


    def evaluate(self, state, target_return=None, pre_reward=None):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim)
            ep_return = target_return if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_rewards[-1] = pre_reward
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long) * self.eval_timesteps[:, -1] + 1], dim=1)
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1)])

        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - self.state_mean) / self.state_std,
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_target_return.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long)
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def take_actions(self, state, target_return=None, pre_reward=None):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            ep_return = target_return if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_rewards[-1] = torch.tensor(pre_reward).to(self.device)
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long) * self.eval_timesteps[:, -1] + 1], dim=1)
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1)])

        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32, device=self.device) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32, device=self.device),
            self.eval_rewards.to(dtype=torch.float32, device=self.device),
            self.eval_target_return.to(dtype=torch.float32, device=self.device),
            self.eval_timesteps.to(dtype=torch.long, device=self.device)
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action



    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32)

        self.eval_target_return = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)

        self.eval_episode_return, self.eval_episode_length = 0, 0

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "dt.pt")
        torch.save(self.state_dict(), file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/dt_model.pth')

    def load_net(self, load_path="saved_model/DTtest", device='cpu'):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device))
        print(f"Model loaded from {self.device}.")
