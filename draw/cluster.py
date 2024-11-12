import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker

source_path = 'log/buffer/Hero-ram-v4True/'
target_path = 'draw/result/cluster/Hero-ram-v4True/'
action=np.load(source_path+'action.npy')
obs=np.load(source_path+'obs.npy')
obs_next=np.load(source_path+'obs_next.npy')
reward=np.load(source_path+'rew.npy')
mask=np.load(source_path+'mask.npy')
update_mask=np.load(source_path+'update_mask.npy')
new_rewards=np.load(source_path+'new_rewards.npy')
mask[mask] = update_mask
reward[mask] = new_rewards

# 定义计算共识矩阵的函数
def consensus_matrix(data, num_clusters, num_iterations, real_labels,method):
    N = data.shape[0]
    consensus = np.zeros((N, N))
    
    # 根据真实标签对数据进行排序
    # sort_idx = np.argsort(real_labels)
    # data = data[sort_idx]
    # real_labels = real_labels[sort_idx]
    
    for _ in range(num_iterations):
        if method == 'KMeans':
            clustering = KMeans(n_clusters=num_clusters, random_state=None).fit(data)
            labels = clustering.labels_
        else:
            clustering = GaussianMixture(n_components=num_clusters, random_state=None).fit(data)
            labels = clustering.predict(data)
        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:
                    consensus[i, j] += 1
    
    consensus /= num_iterations
    
    # 对共识矩阵进行重排
    # consensus = consensus[sort_idx][:, sort_idx]
    return consensus

method = 'GMM'
num_iterations=100

data=np.concatenate((obs, action[:, np.newaxis], obs_next), axis=1)
scaler = StandardScaler()  # 使用标准化
data = scaler.fit_transform(data)  # 标准化数据
print('data shape:',data.shape,'reward shape:',reward.shape)

#生成label
unique_rewards = np.unique(reward[reward != 0])
centers = len(unique_rewards)
print('unique_rewards:',unique_rewards,'centers:',centers)
labels = {value: idx for idx, value in enumerate(unique_rewards)}
reverse_labels = {v: k for k, v in labels.items()}
reward_labels = np.zeros_like(reward)  # 创建与reward形状相同的数组
for value, idx in labels.items():
    reward_labels[reward == value] = idx

#采样
sample_data = []
sample_labels = []
draw_labels = []
for label in np.unique(reward_labels):
    class_data = data[reward_labels == label]
    sample_size = min(class_data.shape[0],20)
    sampled = class_data[np.random.choice(class_data.shape[0], sample_size, replace=False)]
    sample_data.append(sampled)
    sample_labels.extend([label] * sample_size)
    # 只在每个类别的第一个样本显示标签,其余为null
    draw_labels.append([int(reverse_labels[label])] + [None] * (sample_size-1))
    # draw_labels.append([int(reverse_labels[label])] * sample_size)
sample_data = np.vstack(sample_data)  # 合并所有采样的数据
sample_labels = np.array(sample_labels)  # 转换为数组
draw_labels = np.hstack(draw_labels)
print('sample_data shape:',sample_data.shape,'sample_labels shape:',sample_labels.shape, '\nbegin clustering') 

# 计算共识矩阵
consensus = consensus_matrix(sample_data, num_clusters=centers, num_iterations=num_iterations,real_labels=sample_labels,method=method)

# 使用 Seaborn 绘制共识矩阵的热图
plt.figure(figsize=(8, 6))
sns.heatmap(consensus, cmap="viridis", square=True, cbar=True, 
            xticklabels=draw_labels, yticklabels=draw_labels)
# 只显示边界的标签
# # 设置 x 和 y 轴的标签显示间隔
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=10))  # 每隔 5 个显示一次标签
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=10))
# 旋转标签
plt.xticks(rotation=45)
plt.yticks(rotation=45)


plt.title("Consensus Matrix Heatmap")
plt.xlabel("Reward Value")
plt.ylabel("Reward Value") 
plt.savefig(target_path+"consensus_matrix_heatmap_{}_{}.png".format(centers,method))


# # 定义聚类方法及参数
# methods = {
#     "KMeans_k="+str(len(np.unique(sample_labels))): KMeans(n_clusters=len(np.unique(sample_labels))),
#     "DBSCAN": DBSCAN(eps=1 , min_samples=5),
#     "Agglomerative_k="+str(len(np.unique(sample_labels))): AgglomerativeClustering(n_clusters=len(np.unique(sample_labels)))
# }

# # 存储聚类结果及评价指标
# results = []

# # 执行聚类并评估
# for name, model in methods.items():
#     model.fit(sample_data)

#     if hasattr(model, 'labels_'):
#         labels = model.labels_
#     else:
#         labels = model.predict(sample_data)
    
#     ari = adjusted_rand_score(sample_labels, labels)
#     nmi = normalized_mutual_info_score(sample_labels, labels)
#     silhouette = silhouette_score(sample_data, sample_labels)
    
#     results.append({
#         "Method": name,
#         "ARI": ari,
#         "NMI": nmi,
#         "Silhouette Score": silhouette
#     })

# # 展示评价结果
# results_df = pd.DataFrame(results)
# print(results_df)

# # 降维处理，便于可视化
# pca = PCA(n_components=2)
# data_pca = pca.fit_transform(sample_data)

# # 可视化每种聚类结果
# fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))

# # for ax, (name, model) in zip(axes, methods.items()):
# #     if hasattr(model, 'labels_'):
# #         labels = model.labels_
# #     else:
# #         labels = model.predict(sample_data)
    
# #     scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
# #     ax.set_title(name)
# #     legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
# #     ax.add_artist(legend1)

# for ax, (name, model) in zip(axes, methods.items()):
#     if hasattr(model, 'labels_'):
#         labels = model.labels_
#     else:
#         labels = model.predict(sample_data)
    
#     # 通过颜色表示真实标签，不同透明度表示聚类标签
#     for label in np.unique(labels):
#         mask = labels == label
#         ax.scatter(data_pca[mask, 0], data_pca[mask, 1], c=sample_labels[mask], cmap='viridis', 
#                    s=50, alpha=0.4 + 0.4 * (labels[mask] == sample_labels[mask]))  # 增加透明度区分
#     ax.set_title(name)

# plt.suptitle("Clustering Results in 2D Space")
# plt.savefig(target_path+'cluster_result.png')