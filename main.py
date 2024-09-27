import ray
import os
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray import tune, air,train
from Trainer import custom_dqn_train

# 初始化 Ray
ray.init()


tuner = tune.Tuner(
    custom_dqn_train,
    param_space={
        "env": "CartPole-v1",        # 环境
        "framework": "torch",        # 框架
        "num_workers": 1,            # 工作线程
        "num_iterations": 1000,        # 训练迭代次数
        "lr": tune.grid_search([1e-2, 1e-3]),  # Tune 进行学习率搜索
        "gamma": tune.choice([0.95, 0.99]),    # Tune 进行折扣因子的搜索
    },
    run_config=air.RunConfig(
        stop={"training_iteration": 200},
        checkpoint_config=air.CheckpointConfig(num_to_keep=3), 
        storage_path="~/pytorch-ddpg/ray_results"
    )
)

# 运行训练
results = tuner.fit()