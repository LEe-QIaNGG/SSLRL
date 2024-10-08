import gymnasium as gym

# 加载某个 Atari 游戏环境
env = gym.make('CartPole-v1')

# 查看环境的 reward_threshold 属性
print(env.spec.reward_threshold)
