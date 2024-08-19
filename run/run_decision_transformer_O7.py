import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dto7 .utils import EpisodeReplayBufferModify as EpisodeReplayBuffer
from bidding_train_env.baseline.dto7.dt import DecisionTransformer
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle
import torch
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# 获取当前的日期和时间
current_datetime = datetime.now()

# 格式化日期和时间为字符串
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

writer = SummaryWriter(f"results/dto7_{formatted_datetime}")
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

seed = 1000
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
    state_dim = 35

    # replay_buffer = EpisodeReplayBuffer(state_dim, 1, "./bidding_train_env/data/trajectory/trajectory_data_temp_modify.pkl")
    #
    replay_buffer = EpisodeReplayBuffer(state_dim, 1, "./bidding_train_env/data/trajectory/trajectory_data_extended_2_modify.pkl")
    step_num = 400000
    batch_size = 1024 #512 for score 0.35
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std,
                         "state_max": replay_buffer.state_max, "state_min": replay_buffer.state_min},
                        "saved_model/DTO7test")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std, state_max=replay_buffer.state_max, state_min=replay_buffer.state_min)


    model.train()
    model.to(device)
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size
    with open(f'results/dto7_{formatted_datetime}/model_hyperparameters.txt', 'w') as f:
        for key, value in model.hyperparameters.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask, c, ctg in dataloader:
        train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask, c, ctg)
        if i % 1000 == 0:
            logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
            if i % 5000 == 0:
                model.save_net(f"results/dto7_{formatted_datetime}/saved_model/DTO7test/{i}")
                save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std,
                                     "state_max": replay_buffer.state_max, "state_min": replay_buffer.state_min},
                                    f"results/dto7_{formatted_datetime}/saved_model/DTO7test/{i}")

        writer.add_scalar('Action loss', np.mean(train_loss), i)

        model.scheduler.step()
        i += 1

    model.save_net("saved_model/DTO7test")
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
