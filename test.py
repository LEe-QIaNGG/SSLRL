import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
from tianshou.policy import DQNPolicy
from tianshou.data import ReplayBuffer, Collector,PrioritizedReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import OffpolicyTrainer

# 离散化动作空间
class DiscretizedFetchReachEnv(gym.Wrapper):
    def __init__(self, env, n=3):
        super().__init__(env)
        self.n = n  # 动作离散化的程度
        self.action_space = gym.spaces.Discrete(n**3)
        self._discrete_actions = self._create_discrete_actions(n)

    def _create_discrete_actions(self, n):
        actions = []
        for dx in np.linspace(-1, 1, n):
            for dy in np.linspace(-1, 1, n):
                for dz in np.linspace(-1, 1, n):
                    actions.append([dx, dy, dz])
        return np.array(actions)

    def step(self, action):
        continuous_action = self._discrete_actions[action]
        obs, reward, done, truncated, info = self.env.step(continuous_action)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# 创建离散化后的 FetchReach 环境
def make_fetchreach_env(n_envs=1, n_discretize=3):
    def _make_env():
        return DiscretizedFetchReachEnv(gym.make("FetchReach-v3"), n=n_discretize)
    return SubprocVectorEnv([_make_env for _ in range(n_envs)])

# 定义网络
class DQNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(action_shape))
        )

    def forward(self, obs, state=None, info={}):
        obs = torch.tensor(obs, dtype=torch.float32)
        return self.net(obs), state

# 主函数
def main():
    # 超参数
    n_envs = 4  # 并行环境数量
    buffer_size = 50000
    batch_size = 64
    learning_rate = 1e-3
    gamma = 0.99
    target_update_freq = 500
    n_episodes = 100
    exploration_fraction = 0.1
    exploration_final_eps = 0.05

    # 环境
    train_envs = make_fetchreach_env(n_envs=n_envs)
    test_envs = make_fetchreach_env(n_envs=1)

    # 状态和动作维度
    state_shape = train_envs.observation_space["observation"].shape
    action_shape = train_envs.action_space.n

    # 网络
    net = Net(state_shape, action_shape, hidden_sizes=[128, 128], device="cpu")
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # DQN 策略
    policy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        estimation_step=1,
        target_update_freq=target_update_freq,
    )

    # 经验回放
    buffer = PrioritizedReplayBuffer(
            size=30000,
            alpha=0.6, beta=0.4,
            buffer_num=n_envs,
            ignore_obs_next=True,
        )
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # 训练
    result = OffpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=10000,
        step_per_collect=10,
        episode_per_test=10,
        batch_size=batch_size,
        update_per_step=0.1,
        train_fn=lambda epoch, env_step: policy.set_eps(
            max(
                exploration_final_eps,
                exploration_fraction - env_step / (100000 * exploration_fraction),
            )
        ),
        test_fn=lambda epoch, env_step: policy.set_eps(exploration_final_eps),
        stop_fn=lambda mean_rewards: mean_rewards >= -5,  # 停止条件
    )
    print(f"Training Result: {result}")

if __name__ == "__main__":
    main()
