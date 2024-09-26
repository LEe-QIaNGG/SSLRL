from ray.rllib.execution import train_one_step
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
from ReplayBuffer import CustomReplayBuffer
from ray import tune

import torch
import argparse

def dqn_trainer(config):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=50, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=1000, help="Number of timesteps to train."
    )
    args = parser.parse_args()

    custom_replay_buffer_config = {
        "type": CustomReplayBuffer,
        "capacity": 500,  # 设置经验池的容量
    }

    # DQN 算法配置
    config = (
        DQNConfig()
        .environment("CartPole-v1")  # 设置环境
        .framework(framework=args.framework)          # 使用 PyTorch 框架，或者使用 "tf" 对应 TensorFlow
        .env_runners(num_env_runners=1,
                    exploration_config={         
                "epsilon_timesteps": 10000,  # epsilon 从 1.0 衰减到 0.1 的步数
                "final_epsilon": 0.1,        # epsilon 最终值
            })  # 并行 worker 的数量
        .training(
            gamma=0.99,                  # 折扣因子
            lr=1e-3,                     # 学习率
            train_batch_size=32,          # 批量大小
            replay_buffer_config=custom_replay_buffer_config,
            target_network_update_freq=500,  # 目标网络的更新频率
        )
        .resources(num_gpus=0)            # 使用 GPU 的数量，0 表示只使用 CPU
    )


    dqn_algo = config.build()
    workers=dqn_algo.workers

    # 自定义的训练迭代次数
    for i in range(config["stop-iters"]):
        rollouts = synchronous_parallel_sample(workers)
        train_op = rollouts.for_each(
            multi_gpu_train_one_step(
                workers,
                optimizer_fn=lambda w: torch.optim.Adam(w.get_policy().model.parameters(), lr=config["lr"]),
                policies=[DQNTorchPolicy],
                num_gpus=config["num_gpus"],
                train_batch_size=config["train_batch_size"]
            )
        )
        
        # 打印结果
        pretty_print(train_op)
        
        # 向 Tune 上报结果
        tune.report(episode_reward_mean=train_op["episode_reward_mean"])