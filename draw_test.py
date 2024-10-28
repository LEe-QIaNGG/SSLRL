import numpy as np
import matplotlib.pyplot as plt

def draw_distribution(rewards, mask, new_rewards, update_mask):
    mask[mask] = update_mask
    rewards[mask] = new_rewards
    values, counts = np.unique(rewards, return_counts=True)
    plt.bar(values, counts)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution')
    plt.savefig('test.png')
    plt.close()

# file_path = "/home/yangyangjun/lwy/SSLRL/log/reward_distribution/Seaquest-ram-v4/"
# mask = np.load(file_path + "mask_iter_202.npy")
# new_rewards = np.load(file_path + "new_rewards_iter_202.npy")
# rewards = np.load(file_path + "rewards_iter_202.npy")
# update_mask = np.load(file_path + "update_mask_iter_202.npy")
obs = np.load('/home/yangyangjun/lwy/SSLRL/log/buffer/Seaquest-ram-v4/obs.npy')
print(obs.shape)
# draw_distribution(rewards, mask, new_rewards, update_mask)
