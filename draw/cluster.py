import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker

source_path = 'log/buffer/Seaquest-ram-v4/'
target_path = 'draw/result/cluster/Seaquest/'
action = np.load(source_path + 'action.npy')
obs = np.load(source_path + 'obs.npy')
obs_next = np.load(source_path + 'obs_next.npy')
reward = np.load(source_path + 'rew.npy')
mask = np.load(source_path + 'mask.npy')
update_mask = np.load(source_path + 'update_mask.npy')
new_rewards = np.load(source_path + 'new_rewards.npy')
mask[mask] = update_mask
reward[mask] = new_rewards
num_iterations = 100

methods = ['KMeans', 'GMM', 'PCA+KMeans', 'PCA+GMM', 'tSNE+KMeans', 'tSNE+GMM']
consensus_results = {}

# 定义计算共识矩阵的函数
def consensus_matrix(data, num_clusters, num_iterations, real_labels, method):
    N = data.shape[0]
    consensus = np.zeros((N, N))

    for _ in range(num_iterations):
        if method == 'KMeans':
            clustering = KMeans(n_clusters=num_clusters, random_state=None).fit(data)
            labels = clustering.labels_
        elif method == 'GMM':
            clustering = GaussianMixture(n_components=num_clusters, random_state=None).fit(data)
            labels = clustering.predict(data)
        elif method == 'PCA+KMeans':
            pca = PCA(n_components=num_clusters)
            data_transformed = pca.fit_transform(data)
            clustering = KMeans(n_clusters=num_clusters, random_state=None).fit(data_transformed)
            labels = clustering.labels_
        elif method == 'PCA+GMM':
            pca = PCA(n_components=num_clusters)
            data_transformed = pca.fit_transform(data)
            clustering = GaussianMixture(n_components=num_clusters, random_state=None).fit(data_transformed)
            labels = clustering.predict(data_transformed)
        elif method == 'tSNE+KMeans':
            tsne = TSNE(n_components=3)
            data_transformed = tsne.fit_transform(data)
            clustering = KMeans(n_clusters=num_clusters, random_state=None).fit(data_transformed)
            labels = clustering.labels_
        elif method == 'tSNE+GMM':
            tsne = TSNE(n_components=3)
            data_transformed = tsne.fit_transform(data)
            clustering = GaussianMixture(n_components=num_clusters, random_state=None).fit(data_transformed)
            labels = clustering.predict(data_transformed)

        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:
                    consensus[i, j] += 1
    consensus /= num_iterations
    return consensus

data = np.concatenate((obs, action[:, np.newaxis], obs_next), axis=1)
scaler = StandardScaler()  # 使用标准化
data = scaler.fit_transform(data)  # 标准化数据
print('data shape:', data.shape, 'reward shape:', reward.shape)

# 生成label
unique_rewards = np.unique(reward[reward != 0])
centers = len(unique_rewards)
print('unique_rewards:', unique_rewards, 'centers:', centers)
labels = {value: idx for idx, value in enumerate(unique_rewards)}
reverse_labels = {v: k for k, v in labels.items()}
reward_labels = np.zeros_like(reward)  # 创建与reward形状相同的数组
for value, idx in labels.items():
    reward_labels[reward == value] = idx

# 采样
sample_data = []
sample_labels = []
draw_labels = []
for label in np.unique(reward_labels):
    class_data = data[reward_labels == label]
    sample_size = min(class_data.shape[0], 20)
    sampled = class_data[np.random.choice(class_data.shape[0], sample_size, replace=False)]
    sample_data.append(sampled)
    sample_labels.extend([label] * sample_size)
    draw_labels.append([int(reverse_labels[label])] + [None] * (sample_size - 1))
sample_data = np.vstack(sample_data)  # 合并所有采样的数据
sample_labels = np.array(sample_labels)  # 转换为数组
draw_labels = np.hstack(draw_labels)
print('sample_data shape:', sample_data.shape, 'sample_labels shape:', sample_labels.shape, '\nbegin clustering')

# 对每个method计算共识矩阵并保存结果
for method in methods:
    print(f'begin {method}')
    consensus = consensus_matrix(sample_data, num_clusters=centers, num_iterations=num_iterations, real_labels=sample_labels, method=method)
    consensus_results[method] = consensus

    # 使用 Seaborn 绘制共识矩阵的热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(consensus, cmap="viridis", square=True, cbar=True,
                xticklabels=draw_labels, yticklabels=draw_labels)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title(f"Consensus Matrix Heatmap - {method}")
    plt.xlabel("Reward Value")
    plt.ylabel("Reward Value")
    plt.savefig(target_path + f"consensus_matrix_heatmap_{centers}_{method}.png")
