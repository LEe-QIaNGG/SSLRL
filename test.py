from gym import envs

all_envs = envs.registry.keys()

# 打印 Atari 环境
atari_envs = [env_id for env_id in all_envs if 'ALE' in env_id]
print(atari_envs)

# import tianshou
# print(tianshou.__version__)