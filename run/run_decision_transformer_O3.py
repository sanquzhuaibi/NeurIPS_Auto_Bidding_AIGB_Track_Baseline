import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dto3.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dto3.dt import DecisionTransformer
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle
import torch
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# 获取当前的日期和时间
current_datetime = datetime.now()

# 格式化日期和时间为字符串
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

writer = SummaryWriter(f"results/dto3_{formatted_datetime}")
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_dt():
    train_model()


def train_model():
    state_dim = 44

    replay_buffer = EpisodeReplayBuffer(state_dim, 1, "./bidding_train_env/data/trajectory/trajectory_data_temp.csv")
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        "saved_model/DTO3test")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std)
    step_num = 1
    batch_size = 2
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    model.to(device)
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size
    with open(f'results/dto3_{formatted_datetime}/model_hyperparameters.txt', 'w') as f:
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
            if i % 10000 == 0:
                model.save_net(f"results/dto3_{formatted_datetime}/saved_model/DTO3test")

        writer.add_scalar('Action loss', np.mean(train_loss), i)

        model.scheduler.step()
        i += 1

    model.save_net("saved_model/DTO3test")
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/saved_model")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()
