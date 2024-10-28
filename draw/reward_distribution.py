import os
import numpy as np
import matplotlib.pyplot as plt

# 获取当前文件夹下所有 .npy 文件并按文件名排序（假设文件按代命名）
npy_files = sorted([f for f in os.listdir('.') if f.endswith('.npy')])

# 读取每个 .npy 文件的 reward 数据
rewards_per_epoch = [np.load(f) for f in npy_files]

# 绘制 reward 分布图
plt.figure(figsize=(12, 6))

# 使用 boxplot 或 violinplot 显示每代的分布
# 使用 boxplot
plt.boxplot(rewards_per_epoch, positions=range(1, len(rewards_per_epoch) + 1), widths=0.6)

# 使用 violinplot（可选）
# plt.violinplot(rewards_per_epoch, positions=range(1, len(rewards_per_epoch) + 1), widths=0.8, showmeans=True)

# 设置图表标题和标签
plt.title("Reward Distribution per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Reward")

# 美化 x 轴
plt.xticks(range(1, len(rewards_per_epoch) + 1), labels=[f"Epoch {i}" for i in range(1, len(rewards_per_epoch) + 1)])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图表
plt.savefig('reward_distribution.png')
plt.close()
