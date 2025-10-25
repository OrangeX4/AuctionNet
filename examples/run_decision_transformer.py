import numpy as np
import torch
from auctionbid.common.utils import normalize_state, normalize_reward, save_normalize_dict
from auctionbid.baseline.dt.utils import EpisodeReplayBuffer
from auctionbid.baseline.dt.dt import DecisionTransformer
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_dt():
    train_model()


def train_model():
    state_dim = 16

    # replay_buffer = EpisodeReplayBuffer(16, 1, "./data/trajectory/trajectory_data.csv")
    replay_buffer = EpisodeReplayBuffer(16, 1, "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv")
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        "log/DTtest")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std)
    step_num = 10000
    batch_size = 32
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask in dataloader:
        train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        i += 1
        logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
        model.scheduler.step()

    model.save_net("log/DTtest")
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/log/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/log")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    run_dt()
