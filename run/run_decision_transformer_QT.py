# import robohive

# import gym
import numpy as np
import torch
import pandas as pd
import ast
import argparse
import pickle
import random
import sys
import os
import pathlib
import time

# from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from bidding_train_env.baseline.qt.ql_trainer import Trainer
from  bidding_train_env.baseline.qt.ql_DT import DecisionTransformer, Critic
from  bidding_train_env.baseline.qt.logger import logger, setup_logger

# 配置日志
# import logging
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
# )
# logger = logging.getLogger(__name__)
import os
import torch
import random
import time
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# 获取当前的日期和时间
current_datetime = datetime.now()

# 格式化日期和时间为字符串
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

log_path = f"results/qt_{formatted_datetime}"
writer = SummaryWriter(log_path)
def save_normalize_dict(normalize_dict, save_dir):
    """
    Save the normalization dictionary to a Pickle file.

    Args:
        normalize_dict: The dictionary containing normalization statistics.
        save_dir: The directory to save the normalization dictionary.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'normalize_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(normalize_dict, file)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 8  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def save_checkpoint(state, name):
    filename = name
    torch.save(state, filename)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    seed = variant['seed']
    # group_name = f'{exp_prefix}-{env_name}-{dataset}'
    # timestr = time.strftime("%y%m%d-%H%M%S")
    # exp_prefix = f'{group_name}-{seed}-{timestr}'

    scale = 2000
    state_dim = 16
    act_dim = 1
    scale = variant['scale']
    K = variant['K']
    batch_size = variant['batch_size']

    if variant['scale'] is not None:
        scale = variant['scale']

    max_ep_len = variant['max_ep_len']
    variant['env_targets'] = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
    variant['scale'] = scale
    if variant['test_scale'] is None:
        variant['test_scale'] = scale

    if not os.path.exists(os.path.join(variant['save_path'], exp_prefix)):
        pathlib.Path(
            args.save_path +
            exp_prefix).mkdir(
            parents=True,
            exist_ok=True)
    # setup_logger(exp_prefix, variant=variant, log_dir=os.path.join(variant['save_path'], exp_prefix))

    # writer = SummaryWriter(os.path.join(variant['save_path'], exp_prefix))
    # writer = None




    # load dataset

    dataset_path = "./bidding_train_env/data/trajectory/trajectory_data.csv"
    # with open(dataset_path, 'rb') as f:
    #     training_data = pickle.load(f)
    training_data = pd.read_csv(dataset_path)

    # 将参数保存到文本文件
    with open(f'results/qt_{formatted_datetime}/hyperparameters.txt', 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


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
    trajectories = training_data

    states, rewards, actions, returns, traj_lens, dones, next_states = [], [], [], [], [], [], []
    state = []
    reward = []
    action = []
    done = []
    next_state = []
    for index, row in trajectories.iterrows():
        state.append(row["state"])
        reward.append(row['reward_continuous'])
        action.append(row["action"])
        done.append(row["done"])
        if row["done"] == 0:
            next_state.append(row["next_state"])
        else:
            next_state.append((0,)*len(row["state"]))
        if row["done"]:
            if len(state) != 1:
                states.append(np.array(state))
                rewards.append(np.expand_dims(np.array(reward), axis=1))
                actions.append(np.expand_dims(np.array(action), axis=1))
                returns.append(sum(reward))
                traj_lens.append(len(state))
                dones.append(np.array(done))
                next_states.append(np.array(next_state))
            state = []
            reward = []
            action = []
            done = []
            next_state = []
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    tmp_states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-10

    trajectories = []
    for i in range(len(states)):
        # trajectories.append(
        #     {"observations": states[i], "actions": actions[i], "rewards": rewards[i],
        #      "dones": dones[i]})
        trajectories.append(
            {"observations": states[i], "actions": actions[i], "rewards": rewards[i],
             "dones": dones[i], "next_observations": next_states[i]})

    K = K
    pct_traj = 1.

    num_timesteps = sum(traj_lens)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest（从return中最小的值的index排序起来）
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    save_normalize_dict({"state_mean": state_mean, "state_std": state_std},
                        "saved_model/QTtest")
#+++++++++++++++++++++++++++++++++++===================================

    # # save all path information into separate lists
    # mode = variant.get('mode', 'normal')
    # states, traj_lens, returns = [], [], []
    # for path in trajectories:
    #     if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
    #         path['rewards'][-1] = path['rewards'].sum()
    #         path['rewards'][:-1] = 0.
    #     states.append(path['observations'])
    #     traj_lens.append(len(path['observations']))
    #     returns.append(path['rewards'].sum())
    # traj_lens, returns = np.array(traj_lens), np.array(returns)
    #
    # # used for input normalization
    # states = np.concatenate(states, axis=0)
    # state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    #
    # num_timesteps = sum(traj_lens)

    # logger.log('=' * 50)
    # logger.log(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    # logger.log(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    # logger.log(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    # logger.log('=' * 50)

    # K = variant['K']
    # batch_size = variant['batch_size']
    # num_eval_episodes = variant['num_eval_episodes']
    # pct_traj = variant.get('pct_traj', 1.)
    #
    # # only train on top pct_traj trajectories (for %BC experiment)
    # num_timesteps = max(int(pct_traj * num_timesteps), 1)
    # sorted_inds = np.argsort(returns)  # lowest to highest
    # num_trajectories = 1
    # timesteps = traj_lens[sorted_inds[-1]]
    # ind = len(trajectories) - 2
    # while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
    #     timesteps += traj_lens[sorted_inds[ind]]
    #     num_trajectories += 1
    #     ind -= 1
    # sorted_inds = sorted_inds[-num_trajectories:]
    #
    # # used to reweight sampling so we sample according to timesteps instead of trajectories
    # p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    def get_batch(batch_size=256, max_len=K, scale=2000):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask, target_a = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            target_a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1, 1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff

            if variant['reward_tune'] == 'cql_antmaze':
                traj_rewards = (traj['rewards'] - 0.5) * 4.0
            else:
                traj_rewards = traj['rewards']
            r.append(traj_rewards[si:si + max_len].reshape(1, -1, 1))
            rtg.append(discount_cumsum(traj_rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            action_scale = 200
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)/action_scale
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1) / scale
            target_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)/action_scale
            d[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)), d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, target_a, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model, critic):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        critic,
                        max_ep_len=max_ep_len,
                        scale=variant['test_scale'],
                        target_return=[t / variant['test_scale'] for t in target_rew],
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_normalized_score': env.get_normalized_score(np.mean(returns)),
            }

        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=96,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        scale=scale,
        sar=variant['sar'],
        rtg_no_q=variant['rtg_no_q'],
        infer_no_q=variant['infer_no_q']
    )
    critic = Critic(
        state_dim, act_dim, hidden_dim=variant['embed_dim']
    )

    model = model.to(device=device)
    critic = critic.to(device=device)

    trainer = Trainer(
        model=model,
        critic=critic,
        batch_size=batch_size,
        tau=variant['tau'],
        discount=variant['discount'],
        get_batch=get_batch,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None, #[eval_episodes(env_targets)]
        max_q_backup=variant['max_q_backup'],
        eta=variant['eta'],
        eta2=variant['eta2'],
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
        lr_decay=True,#variant['lr_decay']
        lr_maxt=variant['max_iters'],
        lr_min=variant['lr_min'],
        grad_norm=variant['grad_norm'],
        scale=scale,
        k_rewards=True, #variant['k_rewards']
        use_discount=True #variant['use_discount']
    )

    best_ret = -10000
    best_nor_ret = -1000
    best_iter = -1
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], logger=logger,
                                          iter_num=iter + 1, log_writer=writer, log_path=log_path)
        trainer.scale_up_eta(variant['lambda'])
        ret = outputs['Best_return_mean']
        nor_ret = outputs['Best_normalized_score']
        if ret > best_ret:
            state = {
                'epoch': iter + 1,
                'actor': trainer.actor.state_dict(),
                'critic': trainer.critic_target.state_dict(),
            }
            save_checkpoint(state, os.path.join(variant['save_path'], exp_prefix, 'epoch_{}.pth'.format(iter + 1)))
            best_ret = ret
            best_nor_ret = nor_ret
            best_iter = iter + 1
        logger.log(
            f'Current best return mean is {best_ret}, normalized score is {best_nor_ret * 100}, Iteration {best_iter}')

        if variant['early_stop'] and iter >= variant['early_epoch']:
            break
    logger.log(f'The final best return mean is {best_ret}')
    logger.log(f'The final best normalized return is {best_nor_ret * 100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='dt')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=0.)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='./results/')

    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument("--eta2", default=1.0, type=float)
    parser.add_argument("--lambda", default=1.0, type=float)
    parser.add_argument("--max_q_backup", action='store_true', default=False)
    parser.add_argument("--lr_decay", action='store_true', default=False)
    parser.add_argument("--grad_norm", default=2.0, type=float)
    parser.add_argument("--early_stop", action='store_true', default=False)
    parser.add_argument("--early_epoch", type=int, default=100)
    parser.add_argument("--k_rewards", action='store_true', default=False)
    parser.add_argument("--use_discount", action='store_true', default=False)
    parser.add_argument("--sar", action='store_true', default=False)
    parser.add_argument("--reward_tune", default='no', type=str)
    parser.add_argument("--scale", type=float, default=2000)
    parser.add_argument("--test_scale", type=float, default=2000)
    parser.add_argument("--rtg_no_q", action='store_true', default=False)
    parser.add_argument("--infer_no_q", action='store_true', default=False)

    parser.add_argument("--max_ep_len", type=int, default=24)

    args = parser.parse_args()

    experiment(args.exp_name, variant=vars(args))