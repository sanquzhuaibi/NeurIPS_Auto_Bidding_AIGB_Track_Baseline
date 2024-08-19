import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dtiql3.utils import EpisodeReplayBuffer as EpisodeReplayBuffer
from bidding_train_env.baseline.dtiql3.dtiql import DecisionTransformerIQL
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle
import os
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# 获取当前的日期和时间
current_datetime = datetime.now()

# 格式化日期和时间为字符串
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

writer = SummaryWriter(f"results/dtiql3_{formatted_datetime}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


seed = 3407
"""Sets all possible random seeds so results can be reproduced"""
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
# tf.set_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def run_dt():
    train_model()


def train_model():
    state_dim = 16
    replay_buffer = EpisodeReplayBuffer(state_dim, 1, "./bidding_train_env/data/traffic/training_data_rlData_folder/training_data_all-rlData.csv", )
    # replay_buffer = EpisodeReplayBuffer(state_dim, 1, "./bidding_train_env/data/trajectory/trajectory_data_temp.csv",
    #                                     )
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std,
                         "state_max":replay_buffer.state_max, "state_min":replay_buffer.state_min,
                         "reward_max":replay_buffer.reward_max, "reward_min":replay_buffer.reward_min,},
                        "saved_model/DTIQL3test")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DecisionTransformerIQL(dim_obs=state_dim, state_mean=replay_buffer.state_mean, state_std=replay_buffer.state_std,
                                   state_max=replay_buffer.state_max, state_min = replay_buffer.state_min,
                                   reward_max=replay_buffer.reward_max, reward_min=replay_buffer.reward_min)
    step_num = 400000
    batch_size = 128
    model.to(device)
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)#根据这些权重来调整每个样本在训练过程中被抽样的概率，从而更加有效地训练模型。
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size
    with open(f'results/dtiql3_{formatted_datetime}/model_hyperparameters.txt', 'w') as f:
        for key, value in model.hyperparameters.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask, next_state in dataloader:
        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_state, dones, rtg, timesteps, attention_mask)
        # logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")

        writer.add_scalar('Q_loss', q_loss, i)
        writer.add_scalar('V_loss', v_loss, i)
        writer.add_scalar('A_loss', a_loss, i)

        if i % 1000 == 0:
            logger.info(f'Step: {i} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')
            if i % 5000 == 0:
                model.save_net(f"results/dtiql3_{formatted_datetime}/saved_model/DTIQL3test/{i}")
                # save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std,
                #                      "state_max": replay_buffer.state_max, "state_min": replay_buffer.state_min},
                #                     f"results/dtiql3_{formatted_datetime}/saved_model/DTIQL3test/{i}")
                save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std,
                                     "state_max": replay_buffer.state_max, "state_min": replay_buffer.state_min,
                                     "reward_max": replay_buffer.reward_max, "reward_min": replay_buffer.reward_min},
                                    f"results/dtiql3_{formatted_datetime}/saved_model/DTIQL3test/{i}")
        i += 1

        # model.vascheduler.step()

        model.value_scheduler.step()
        model.critic1_scheduler.step()
        model.critic2_scheduler.step()
        model.actor_scheduler.step()

    model.save_net("saved_model/DTIQL3test")

    # model.save_jit('saved_model/DTIQL3test')
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.actors.take_actions(test_state)}")


def load_model():
    """
    加载模型。
    """
    with open('./Model/DTIQL/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformerIQL(dim_obs=16, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DTIQL/saved_model")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.actors.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()
