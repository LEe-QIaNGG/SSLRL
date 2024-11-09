import numpy as np
source_path = '../log/reward_distribution/Hero-ram-v41/'
target_path='../draw/draw_source/reward_distribution/Hero-ram-v4/L2False/'
for i in range(1,6):
    iter=i*40000
    mask=np.load(source_path+'mask_iter_{}.npy'.format(iter))
    reward=np.load(source_path+'rewards_iter_{}.npy'.format(iter))
    update_mask = np.load(source_path+'update_mask_iter_{}.npy'.format(iter))
    new_rewards = np.load(source_path+'new_rewards_iter_{}.npy'.format(iter))
    mask[mask] = update_mask
    print(np.unique(reward[reward != 0]))
    reward[mask] = new_rewards
    print(np.unique(reward[reward != 0]))
    # np.save(target_path+'reward_{}.npy'.format(iter//200),reward)