class RewardEstimator:
    def __init__(self, max_reward, min_reward, num_reward=10):
        step = (max_reward - min_reward) / (num_reward - 1)
        self.reward_values = [round(min_reward + i * step) for i in range(num_reward)]
        self.num_reward = len(self.reward_values)

    def estimate_reward(self, samples):
        rewards = samples["rewards"]
        constant = 1.0  # ����һ������ֵ�����Ը�����Ҫ����
        
        # �Է���Ľ������ϳ���
        modified_rewards = [r + constant if r != 0 else r for r in rewards]
        
        # ���� SampleBatch �еĽ���
        samples["rewards"] = modified_rewards
        
        return samples
        