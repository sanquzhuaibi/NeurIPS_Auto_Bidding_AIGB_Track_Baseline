import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


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
        self.register_buffer("masked_bias", torch.tensor(-1e4).cuda())

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

    def __init__(self, state_dim, act_dim, state_mean, state_std, state_max, state_min, action_tanh=False, K=10, max_ep_len=96, scale=2000,
                 target_return=100):
        super(DecisionTransformer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(3407)

        self.length_times = 7
        self.hidden_size = 512
        self.state_mean = state_mean
        self.state_std = state_std
        self.state_max = state_max
        self.state_min = state_min
        # assert self.hidden_size == config['n_embd']
        self.max_length = K
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return

        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = 1e-4


        self.block_config = {
            "n_ctx": 1024,
            "n_embd": self.hidden_size ,
            "n_layer": 3,
            "n_head": 8,
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
            "state_max": self.state_max,
            "state_min": self.state_min,
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
        self.embed_cost = torch.nn.Linear(1, self.hidden_size)
        self.embed_ctg = torch.nn.Linear(1, self.hidden_size)
        self.embed_coef = torch.nn.Linear(1, self.hidden_size)
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

    def forward(self, states, actions, rewards, returns_to_go, costs, ctg,  coef, return_o, return_c,timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)
        costs_embeddings = self.embed_cost(costs)
        ctg_embeddings = self.embed_ctg(ctg)
        coef_embeddings = self.embed_coef(coef)
        return_o_embeddings = self.embed_coef(return_o)
        return_c_embeddings = self.embed_coef(return_c)



        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings


        costs_embeddings = costs_embeddings + time_embeddings
        ctg_embeddings = ctg_embeddings + time_embeddings
        coef_embeddings = coef_embeddings + time_embeddings
        return_o_embeddings = return_o_embeddings + time_embeddings
        return_c_embeddings = return_c_embeddings + time_embeddings

        rewards_embeddings = rewards_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (return_o_embeddings, return_c_embeddings, ctg_embeddings, returns_embeddings, coef_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.length_times * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)

        # return_preds = self.predict_return(x[:, 2])
        # state_preds = self.predict_state(x[:, 2])
        # action_preds = self.predict_action(x[:, 1])

        return_preds = self.predict_return(x[:, 6])
        state_preds = self.predict_state(x[:, 6])
        action_preds = self.predict_action(x[:, 5])
        return state_preds, action_preds, return_preds, None

    def get_action(self, states, actions, rewards, returns_to_go, costs, ctg, coef, return_o, return_c, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        costs = costs.reshape(1, -1, 1)
        costs_to_go = ctg.reshape(1, -1, 1)
        coef = coef.reshape(1, -1, 1)
        return_o = return_o.reshape(1, -1, 1)
        return_c = return_c.reshape(1, -1, 1)

        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            costs_to_go = costs_to_go[:, -self.max_length:]
            costs = costs[:, -self.max_length:]
            coef = coef[:, -self.max_length:]
            return_o = return_o[:, -self.max_length:]

            return_c = return_c[:, -self.max_length:]

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
            costs_to_go = torch.cat(
                [torch.zeros((costs_to_go.shape[0], self.max_length - costs_to_go.shape[1], 1),
                             device=costs_to_go.device), costs_to_go],
                dim=1).to(dtype=torch.float32)
            costs = torch.cat(
                [torch.zeros((costs.shape[0], self.max_length - costs.shape[1], 1), device=costs.device),
                 costs],
                dim=1).to(dtype=torch.float32)
            coef = torch.cat(
                [torch.zeros((coef.shape[0], self.max_length - coef.shape[1], 1), device=coef.device),
                 coef],
                dim=1).to(dtype=torch.float32)
            return_o = torch.cat(
                [torch.zeros((return_o.shape[0], self.max_length - return_o.shape[1], 1), device=return_o.device),
                 return_o],
                dim=1).to(dtype=torch.float32)
            return_c = torch.cat(
                [torch.zeros((return_c.shape[0], self.max_length - return_c.shape[1], 1), device=return_c.device),
                 return_c],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, reward_preds = self.forward(
            states, actions, rewards, returns_to_go, costs, costs_to_go, coef, return_o, return_c, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0, -1]

    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask, costs, ctg, coef, return_o, return_c):
        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        dones = dones.cuda()
        rtg = rtg.cuda()
        timesteps = timesteps.cuda()
        attention_mask = attention_mask.cuda()
        costs = costs.cuda()
        ctg = ctg.cuda()
        coef = coef.cuda()
        return_o = return_o.cuda()
        return_c = return_c.cuda()


        # rewards_target, action_target, rtg_target = torch.clone(rewards), torch.clone(actions), torch.clone(rtg)
        cost_target, rewards_target, action_target, rtg_target = torch.clone(costs), torch.clone(rewards), torch.clone(actions), torch.clone(rtg)

        state_preds, action_preds, return_preds, reward_preds = self.forward(
            states, actions, rewards, rtg[:, :-1], costs, ctg[:, :-1], coef, return_o, return_c, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()

    def take_actions(self, state, target_return=None, pre_reward=None, pre_cost=None):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim)
            self.eval_states = self.eval_states.cuda()
            ep_return = target_return if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1)
            self.eval_target_return = self.eval_target_return.cuda()

            self.eval_target_cost_to_go = torch.tensor(-2000, dtype=torch.float32).reshape(1, 1)
            self.eval_target_cost_to_go = self.eval_target_cost_to_go.cuda()

            self.eval_target_coef = torch.tensor(1, dtype=torch.float32).reshape(1, 1)
            self.eval_target_coef = self.eval_target_coef.cuda()
            self.eval_target_return_o = self.eval_target_return
            self.eval_target_return_c = self.eval_target_return

        else:
            assert pre_reward is not None
            assert pre_cost is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim)
            cur_state = cur_state.cuda()
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)

            self.eval_rewards[-1] = torch.tensor(pre_reward).cuda()
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)

            self.eval_costs[-1] = torch.tensor(pre_cost).cuda()
            pred_cost_to_go = self.eval_target_cost_to_go[0, -1] - (pre_cost / self.scale)
            self.eval_target_cost_to_go = torch.cat([self.eval_target_cost_to_go, pred_cost_to_go.reshape(1, 1)], dim=1)

            self.eval_target_coef = torch.cat([self.eval_target_coef, torch.ones(1,1).cuda()], dim=1)
            self.eval_target_return_o = torch.cat([self.eval_target_return_o, torch.tensor(self.eval_target_return_o[0,0]).reshape(1, 1).cuda()], dim=1)
            self.eval_target_return_c = self.eval_target_return_o


            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long).cuda() * self.eval_timesteps[:, -1] + 1], dim=1)
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1)])
        self.eval_costs = torch.cat([self.eval_costs, torch.zeros(1)])

        # action = self.get_action(
        #     self.eval_states.to(dtype=torch.float32).cuda()  - torch.tensor(self.state_mean).cuda() / torch.tensor(self.state_std).cuda(),
        #     self.eval_actions.to(dtype=torch.float32).cuda() ,
        #     self.eval_rewards.to(dtype=torch.float32).cuda() ,
        #     self.eval_target_return.to(dtype=torch.float32).cuda() ,
        #     self.eval_timesteps.to(dtype=torch.long).cuda()
        # )
        action = self.get_action(
            ( self.eval_states.to(dtype=torch.float32).cuda() - torch.tensor(self.state_min).cuda()) / (torch.tensor(self.state_max).cuda()  - torch.tensor(self.state_min).cuda()+1e-10),
            self.eval_actions.to(dtype=torch.float32).cuda() ,
            self.eval_rewards.to(dtype=torch.float32).cuda() ,
            self.eval_target_return.to(dtype=torch.float32).cuda() ,
            self.eval_costs.to(dtype=torch.float32).cuda(),
            self.eval_target_cost_to_go.to(dtype=torch.float32).cuda(),
            self.eval_target_coef.to(dtype=torch.float32).cuda() ,#望望coef一直是1
            self.eval_target_return_o.to(dtype=torch.float32).cuda(),  # 望望
            self.eval_target_return_c.to(dtype=torch.float32).cuda(),  # 望望

            self.eval_timesteps.to(dtype=torch.long).cuda()
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32)
        self.eval_costs = torch.zeros(0, dtype=torch.float32)

        self.eval_target_return = None
        self.eval_target_cost_to_go = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)
        self.eval_timesteps = self.eval_timesteps.cuda()

        self.eval_episode_return, self.eval_episode_length = 0, 0
        self.eval_episode_cost_to_go = 0


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
