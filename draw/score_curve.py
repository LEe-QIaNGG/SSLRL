import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# 定义颜色列表
colors = ['blue', 'red', 'green', 'orange', 'purple']

# 定义源目录
source_dir = "log/Seaquest-ram-v4/framework_test/"

# 查找所有的 events 文件
event_files = glob(os.path.join(source_dir, "**/events.out.tfevents.*"), recursive=True)

plt.figure(figsize=(12, 8))

for i, event_file in enumerate(event_files):
    steps = []
    best_rewards = []
    best_reward_stds = []

    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag == "info/best_reward":
                steps.append(event.step)
                best_rewards.append(value.simple_value)
            elif value.tag == "info/best_reward_std":
                best_reward_stds.append(value.simple_value)

    # 绘制线条和浅色区域
    folder_name = os.path.basename(os.path.dirname(event_file))
    color = colors[i % len(colors)]
    plt.plot(steps, best_rewards, label=folder_name, color=color)
    plt.fill_between(steps, 
                     np.array(best_rewards) - np.array(best_reward_stds), 
                     np.array(best_rewards) + np.array(best_reward_stds), 
                     alpha=0.2, color=color)

plt.xlabel('训练步数')
plt.ylabel('最佳奖励')
plt.title('不同训练配置的最佳奖励对比')
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig('reward_distribution.png')
plt.close()

print("图像已保存为 reward_distribution.png")
