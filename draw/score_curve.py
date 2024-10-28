import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# 定义颜色列表
colors = ['blue', 'red', 'green', 'orange', 'purple']

# 定义源目录
source_dir = "draw/draw_source/framework/Seaquest/"

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

    # 应用移动平均平滑最佳奖励
    window_size = 900  # 窗口大小，可以根据需要调整

    # 绘制线条和浅色区域
    folder_name = os.path.basename(os.path.dirname(event_file))
    color = colors[i % len(colors)]
    
    import pandas as pd

    # 将数据转为 pandas.Series 对象，方便处理
    rewards_series = pd.Series(best_rewards)
    smoothed_rewards = rewards_series.rolling(window=window_size, min_periods=1, center=False).mean().to_numpy()

    # 绘制完整数据的平滑曲线
    plt.plot(steps, smoothed_rewards, label=folder_name.split('24')[0], color=color)
    plt.fill_between(steps, 
                    smoothed_rewards - np.array(best_reward_stds), 
                    smoothed_rewards + np.array(best_reward_stds), 
                    alpha=0.2, color=color)


plt.xlabel('episode')
plt.ylabel('Seaquest-ram-v4')
plt.title('best reward of different training configurations')
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig('draw/best_reward.png')
plt.close()

print("图像已保存为 best_reward.png")
