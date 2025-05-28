import numpy as np
import matplotlib.pyplot as plt

# dir1 = "log/monitor/update_num/update_num.npy"
# dir2 = "log/monitor/nonzero_num/nonzero_num.npy"
# update_num = np.load(dir1)
# print(update_num)
# nonzero_num = np.load(dir2)  # 读取非零计数数据

# 创建x轴数据点
# x = np.arange(len(update_num))

# # 绘制折线图
# plt.figure(figsize=(10, 6))
# plt.plot(x, update_num, '-', linewidth=1, label='Update Count')
# # plt.plot(x, nonzero_num, '-', linewidth=1, label='Nonzero Count')  # 绘制非零计数
# plt.xlabel('Iteration')
# plt.ylabel('Count')
# plt.title('Update Count vs. Iteration')
# plt.legend()  # 添加图例
# plt.grid(True)
# plt.savefig('./update_num.png')

import matplotlib.pyplot as plt
import os
from ale_py import ALEInterface
ale = ALEInterface()
import gymnasium as gym
from gymnasium import envs
for env in envs.registry:
        print(env)
env = gym.make('Venture-ram-v4')


# def render_robotics_env(env_name="FetchReach-v3", num_frames=5, save_dir="renders"):
#         # 创建环境
#         env = gym.make(env_name, render_mode="rgb_array")
#         # env = gym.make(env_name)
#         obs = env.reset()

#         # 创建保存图片的目录
#         os.makedirs(save_dir, exist_ok=True)

#         for frame_idx in range(num_frames):
#                 # 随机选择动作（可改为特定策略）
#                 action = env.action_space.sample()

#                 # 与环境交互
#                 # obs, reward, done, info = env.step(action)
#                 env.step(action)

#                 # 渲染当前帧
#                 frame = env.render()

#                 # 保存图片
#                 plt.imshow(frame)
#                 plt.axis("off")
#                 file_path = os.path.join(save_dir, f"frame_{frame_idx}.png")
#                 plt.savefig(file_path)
#                 print(f"Saved frame {frame_idx} to {file_path}")

#                 # if done:
#                 #         obs = env.reset()
 
#         env.close()
# render_robotics_env(env_name="FetchReach-v3", num_frames=5, save_dir="renders")