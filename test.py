import numpy as np
import matplotlib.pyplot as plt

dir="log/update_num/FetchReach-v3/False/update_num.npy"
update_num = np.load(dir)

# 创建x轴数据点
x = np.arange(len(update_num))

# 绘制折线图
plt.figure(figsize=(10,6))
plt.plot(x, update_num, '-', linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Update Count')
plt.title('Update Count vs. Iteration')
plt.grid(True)
plt.savefig('./update_num.png')