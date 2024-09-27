from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import air
from ray import tune

config = DQNConfig()
config = config.training(
    num_atoms=tune.grid_search([1,]),  # 增加了更多的离散支持点选项
    gamma=tune.uniform(0.95, 0.99),  # 添加折扣因子搜索
    lr=tune.loguniform(1e-4, 1e-2),  # 将学习率搜索移到training配置中
)
config = config.environment(env="CartPole-v1")

tuner = tune.Tuner(
    "DQN",
    run_config=air.RunConfig(
        stop={
            "episode_reward_mean": 195,  # 当平均奖励达到195时停止
            "training_iteration": 200,  # 最多训练200次迭代
        },
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=10,  # 每10次迭代保存一次检查点
            num_to_keep=5,  # 保留最后5个检查点
        ),
        storage_path="~/pytorch-ddpg/ray_results"
    ),
    param_space=config.to_dict()
)

results = tuner.fit()

# 打印最佳结果
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
print(f"最佳训练结果: {best_result.metrics}")
print(f"最佳配置: {best_result.config}")