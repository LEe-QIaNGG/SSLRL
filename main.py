import ray
import os
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray import tune, air,train
from ReplayBuffer import CustomReplayBuffer


def custom_execution_plan(workers, dqn_algo):
    # # 1. 从 workers 中并行采样数据
    train_batch = synchronous_parallel_sample(worker_set=dqn_algo.env_runner_group)
    # 2. Updating the Policy.
    train_results = train_one_step(dqn_algo, train_batch)
    # 3. Synchronize worker weights.
    dqn_algo.env_runner_group.sync_weights()
    
    
    return train_results

# 自定义 DQN 训练过程
def custom_dqn_train(config):
    custom_replay_buffer_config = {
        "type": CustomReplayBuffer,
        "capacity": 50000,  # 设置经验池的容量
    }
    # 创建 DQN 配置
    dqn_config = DQNConfig().environment(config["env"]).framework(config["framework"]).training(
        gamma=config["gamma"], lr=config["lr"], replay_buffer_config=custom_replay_buffer_config,
    )

    # 构建算法对象
    dqn_algo = dqn_config.build()

    # 获取 workers
    workers = dqn_algo.workers
    
    # 使用自定义的执行计划进行训练
    for i in range(config["num_iterations"]):
        # 执行一次自定义训练（采样+更新）
        results = custom_execution_plan(workers, dqn_algo).__iter__().__next__()

         # 向 Tune 上报结果
        train.report({'score':results})
    # 手动保存最后的 checkpoint
    checkpoint_dir = train.get_context().get_trial_dir()# 获取存储检查点的目录
    checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint")
    dqn_algo.save(checkpoint_path)
    train.report({'score':results}, checkpoint=checkpoint_path)

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