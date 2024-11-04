import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

source_path = 'draw/draw_source/reward_distribution/Hero-ram-v4/L2true1/'
npy_files = sorted([f for f in os.listdir(source_path) if f.endswith('.npy')])

# 读取每个 .npy 文件的 reward 数据
# 修改为使用完整路径加载文件
rewards_per_epoch = [np.load(os.path.join(source_path, f)) for f in npy_files]
reward_counts_per_epoch = [np.unique(reward, return_counts=True) for reward in rewards_per_epoch]

def smooth_frequency(frequency, factor=0.5):
    return frequency ** factor  # 平方根缩放，将 factor 设为 0.5

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline


# reward_counts_per_epoch[0][1][0] = np.sqrt(reward_counts_per_epoch[0][1][0])
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 对每个 epoch 的 reward_counts_per_epoch 进行绘图
for i, (rewards, counts) in enumerate(reward_counts_per_epoch):
    # 归一化频数
    counts_normalized = counts / np.sum(counts)
    counts_normalized = smooth_frequency(counts_normalized)
    # counts_normalized = counts_normalized/np.max(counts_normalized)
    print(counts_normalized)

    
    # 使用 y 值表示不同的 epoch
    y_values = np.full_like(rewards, i)  # 每个 epoch 的 y 值为其索引

    # 绘制 3D 曲线
    ax.bar3d(rewards, y_values, np.zeros_like(counts_normalized), 
             dx=0.1, dy=0.1, dz=counts_normalized,  # 添加 dx, dy, dz 参数
             label=f"Epoch {(i+1)*200}", alpha=0.5)

# 添加标签和标题
ax.set_xlabel("Reward Value")
ax.set_ylabel("Epoch")
ax.set_zlabel("Normalized Frequency")
ax.set_title("3D Distribution of Rewards Over Epochs")
ax.legend()

plt.savefig(source_path+'reward_distribution_3d.png')


