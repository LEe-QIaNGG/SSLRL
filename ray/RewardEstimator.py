class RewardEstimator:
    def __init__(self, max_reward, min_reward, num_reward=10):
        step = (max_reward - min_reward) / (num_reward - 1)
        self.reward_values = [round(min_reward + i * step) for i in range(num_reward)]
        self.num_reward = len(self.reward_values)

    def estimate_reward(self, samples):
        rewards = samples["rewards"]
        constant = 1.0  # 设置一个常数值，可以根据需要调整
        
        # 对非零的奖励加上常数
        modified_rewards = [r + constant if r != 0 else r for r in rewards]
        
        # 更新 SampleBatch 中的奖励
        samples["rewards"] = modified_rewards
        
        return samples
        