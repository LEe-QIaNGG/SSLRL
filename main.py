import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
import gym

# 启动Ray
ray.init()

# 注册环境
def atari_env_creator(env_config):
    env = gym.make('PongNoFrameskip-v4')  # 选择你想要的Atari游戏
    env = wrap_deepmind(env, frame_stack=True)  # 使用DeepMind包装器进行预处理
    return env

tune.register_env("atari_env", atari_env_creator)

# 配置DQN算法
config = {
    "env": "atari_env",
    "framework": "torch",  # 或者 "torch" 来使用 PyTorch
    "num_gpus": 0,  # 如果有GPU可以设置为1
    "num_workers": 3,  # 并行环境工作者的数量，越高越能加速训练
    "train_batch_size": 32,
    "exploration_config": {
        "epsilon_timesteps": 1000000,  # 探索的时间步长
        "final_epsilon": 0.01,  # 最终的epsilon值
    },
    "dueling": True,  # 使用Dueling DQN
    "double_q": True,  # 使用Double Q-Learning
    "prioritized_replay": True,  # 使用优先经验回放
    "lr": 1e-4,  # 学习率
    "gamma": 0.99,  # 折扣因子
    "buffer_size": 50000,  # 经验回放缓冲区大小
    "learning_starts": 10000,  # 多少步之后开始学习
    "target_network_update_freq": 500,  # 目标网络的更新频率
}

# 开始训练
tune.run(DQNTrainer, config=config)
