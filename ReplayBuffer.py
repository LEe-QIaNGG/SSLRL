# -*- coding: utf-8 -*-
import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils.replay_buffers import ReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch
import random

# 自定义 ReplayBuffer
class CustomReplayBuffer(ReplayBuffer):
    def estimateReward(self):
       print(self.sample(1))