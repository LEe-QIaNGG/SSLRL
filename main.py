# -*- coding: utf-8 -*-

import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig

# 初始化 Ray
ray.init()

# DQN 算法配置
config = (
    DQNConfig()
    .environment("CartPole-v1")  # 设置环境
    .framework("torch")          # 使用 PyTorch 框架，或者使用 "tf" 对应 TensorFlow
    .env_runners(num_env_runners=1,
                 exploration_config={         
            "epsilon_timesteps": 10000,  # epsilon 从 1.0 衰减到 0.1 的步数
            "final_epsilon": 0.1,        # epsilon 最终值
        })  # 并行 worker 的数量
    .training(
        gamma=0.99,                  # 折扣因子
        lr=1e-3,                     # 学习率
        train_batch_size=32,          # 批量大小
        replay_buffer_config={
            "capacity": 50000         # 回放缓冲区大小
        },
        target_network_update_freq=500,  # 目标网络的更新频率
    )
    .resources(num_gpus=0)            # 使用 GPU 的数量，0 表示只使用 CPU
)

# 运行训练
tuner = tune.Tuner(
    "DQN",
    param_space=config.to_dict(),      # 将配置转为字典形式传递
    run_config=air.RunConfig(stop={"episode_reward_mean": 200}, 
                             checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True      # 使用新的 CheckpointConfig 管理检查点
        ))
)

results = tuner.fit()

# 获取并打印最优配置
best_result = results.get_best_result()
print("Best config: ", best_result.config)

# 关闭 Ray
ray.shutdown()
