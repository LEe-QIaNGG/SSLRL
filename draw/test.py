import numpy as np
import os

# 定义源目录和目标文件名
source_dir = 'draw/draw_source/reward_distribution/Seaquest-ram-v4'
output_file = 'draw/draw_source/reward_distribution/Seaquest-ram-v4/rewards_epoch_1000.npy'

# 存储所有读取的数组
all_rewards = []

# 遍历源目录下的所有文件夹
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file == 'rewards_epoch_1000.npy':
            file_path = os.path.join(root, file)
            # 读取npy文件并添加到列表中
            rewards = np.load(file_path)
            all_rewards.append(rewards)

# 合并所有数组
merged_rewards = np.concatenate(all_rewards)

# 保存合并后的数组到新的npy文件
np.save(output_file, merged_rewards)

print(f"合并完成，文件保存为: {output_file}")