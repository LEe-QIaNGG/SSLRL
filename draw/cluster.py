import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

source_path = 'draw/draw_source/cluster/Hero-ram-v41/'
target_path = 'draw/result/cluster/'
action=np.load(source_path+'action.npy')
obs=np.load(source_path+'obs.npy')
obs_next=np.load(source_path+'obs_next.npy')
reward=np.load(source_path+'rew.npy')
mask=np.load(source_path+'mask.npy')
update_mask=np.load(source_path+'update_mask.npy')
new_rewards=np.load(source_path+'new_rewards.npy')
mask[mask] = update_mask
reward[mask] = new_rewards
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data=np.concatenate((obs, action[:, np.newaxis], obs_next), axis=1)
scaler = StandardScaler()  # 使用标准化
data = scaler.fit_transform(data)  # 标准化数据
print(data.shape,reward.shape)

#生成label
unique_rewards = np.unique(reward[reward != 0])
print(unique_rewards)
labels = {value: idx for idx, value in enumerate(unique_rewards)}
reward_labels = np.zeros_like(reward)  # 创建与reward形状相同的数组
for value, idx in labels.items():
    reward_labels[reward == value] = idx

#采样
sample_data = []
sample_labels = []  
for label in np.unique(reward_labels):
    class_data = data[reward_labels == label]
    sample_size = max(1, int(0.02 * class_data.shape[0]))  # 确保至少采样一个样本
    sampled = class_data[np.random.choice(class_data.shape[0], sample_size, replace=False)]
    sample_data.append(sampled)
    sample_labels.extend([label] * sample_size)  
sample_data = np.vstack(sample_data)  # 合并所有采样的数据
sample_labels = np.array(sample_labels)  # 转换为数组
print(len(sample_data),len(np.unique(sample_labels)), '\nbegin clustering')


# 定义聚类方法及参数
methods = {
    "KMeans_k="+str(len(np.unique(sample_labels))): KMeans(n_clusters=len(np.unique(sample_labels))),
    "DBSCAN": DBSCAN(eps=1 , min_samples=5),
    "Agglomerative_k="+str(len(np.unique(sample_labels))): AgglomerativeClustering(n_clusters=len(np.unique(sample_labels)))
}

# 存储聚类结果及评价指标
results = []

# 执行聚类并评估
for name, model in methods.items():
    model.fit(sample_data)

    if hasattr(model, 'labels_'):
        labels = model.labels_
    else:
        labels = model.predict(sample_data)
    
    ari = adjusted_rand_score(sample_labels, labels)
    nmi = normalized_mutual_info_score(sample_labels, labels)
    silhouette = silhouette_score(sample_data, sample_labels)
    
    results.append({
        "Method": name,
        "ARI": ari,
        "NMI": nmi,
        "Silhouette Score": silhouette
    })

# 展示评价结果
results_df = pd.DataFrame(results)
print(results_df)

# 降维处理，便于可视化
pca = PCA(n_components=2)
data_pca = pca.fit_transform(sample_data)

# 可视化每种聚类结果
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))

# for ax, (name, model) in zip(axes, methods.items()):
#     if hasattr(model, 'labels_'):
#         labels = model.labels_
#     else:
#         labels = model.predict(sample_data)
    
#     scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
#     ax.set_title(name)
#     legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
#     ax.add_artist(legend1)

for ax, (name, model) in zip(axes, methods.items()):
    if hasattr(model, 'labels_'):
        labels = model.labels_
    else:
        labels = model.predict(sample_data)
    
    # 通过颜色表示真实标签，不同透明度表示聚类标签
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(data_pca[mask, 0], data_pca[mask, 1], c=sample_labels[mask], cmap='viridis', 
                   s=50, alpha=0.4 + 0.4 * (labels[mask] == sample_labels[mask]))  # 增加透明度区分
    ax.set_title(name)

plt.suptitle("Clustering Results in 2D Space")
plt.savefig(target_path+'cluster_result.png')