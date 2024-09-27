# -*- coding: utf-8 -*-
import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils.replay_buffers import ReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch
import random
from RewardEstimator import RewardEstimator
# 自定义 ReplayBuffer
class CustomReplayBuffer(ReplayBuffer):
    def __init__(self, *args, init_sample_size=2000, update_frequency=2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_estimator = None
        self.rewards = []
        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        self.num_reward = 10
        self.init_sample_size = init_sample_size  # 增加初始化样本大小
        self.update_frequency = update_frequency  # 添加更新频率
        self.sample_count = 0

    def add(self, obs, action, reward, next_obs, done, infos):
        super().add(obs, action, reward, next_obs, done, infos)
        
        self.rewards.append(reward)
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        self.sample_count += 1

        # 只有在收集了足够的样本后才初始化或更新 RewardEstimator
        if self.sample_count >= self.init_sample_size and self.sample_count % self.update_frequency == 0:
            if self.reward_estimator is None:
                self._initialize_reward_estimator()
            else:
                self._update_reward_estimator()

    def _initialize_reward_estimator(self):
        print(f"初始化 最大奖励: {self.max_reward} 最小奖励: {self.min_reward}")
        self.reward_estimator = RewardEstimator(
            self.max_reward, 
            self.min_reward, 
            num_reward=self.num_reward
        )

    def estimateReward(self):
       print(self.sample(1))

