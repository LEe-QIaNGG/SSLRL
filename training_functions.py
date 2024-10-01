# -*- coding: utf-8 -*-
import torch

class Net(torch.nn.Module):
    def __init__(self, num_reward):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.fc2 = torch.nn.Linear(10, num_reward)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Reward_Estimator:
    def __init__(self):
        '''要求环境的action是discrete
        可以尝试的几种增强方式：
        1. 保持顺序smooth 

        '''
        self.net = Net()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.reward_list=[-2,-1,0,1,2]
        self.threshold=0.7

    def get_input_data(self, buffer, mask):
        obs = buffer.obs[mask]
        action = buffer.act[mask]
        next_obs = buffer.obs_next[mask]
        return torch.cat([obs, next_obs, action], dim=-1)
    
    def weak_augment(self, input_data):
        # 分离action列
        data_without_action = input_data[:, :-1]
        action = input_data[:, -1:]
        
        # 对除action外的数据添加高斯噪声
        noise = torch.randn_like(data_without_action) * 0.1  # 0.1是噪声强度，可以根据需要调整
        augmented_data = data_without_action + noise
        
        # 重新组合数据
        return torch.cat([augmented_data, action], dim=-1)
    
    def strong_augment(self, input_data, n=3):
        # 分离action列
        data_without_action = input_data[:, :-1]
        action = input_data[:, -1:]
        
        # 对除action外的数据进行n条数据间的平滑操作
        smoothed_data = torch.zeros_like(data_without_action)
        for i in range(data_without_action.shape[1]):
            start = max(0, i - n // 2)
            end = min(data_without_action.shape[1], i + n // 2 + 1)
            smoothed_data[:, i] = torch.mean(data_without_action[:, start:end], dim=1)
        
        # 重新组合数据
        return torch.cat([smoothed_data, action], dim=-1)

    def update_network(self, buffer, alpha):
        # 分别计算非零奖励和零奖励的损失
        mask_nonzero = buffer.rew != 0
        mask_zero = buffer.rew == 0

        # # 计算真实奖励的损失
        input_data_nonzero = self.get_input_data(buffer, mask_nonzero)
        confidence_scores = self.net(input_data_nonzero)
        max_confidence, max_indices = torch.max(confidence_scores, dim=1)
        predicted_reward = torch.tensor([self.reward_list[i] if conf > self.threshold else 0 for i, conf in zip(max_indices, max_confidence)])
        loss_nonzero = torch.nn.MSELoss()(predicted_reward, buffer.rew[mask_nonzero])
        
        input_data_zero = self.get_input_data(buffer, mask_zero)
        input_data_zero_weak = self.weak_augment(input_data_zero)
        input_data_zero_strong = self.strong_augment(input_data_zero)
        combined_input = torch.cat([input_data_zero_weak, input_data_zero_strong], dim=0)
        combined_confidence_scores = self.net(combined_input)
        batch_size = input_data_zero_weak.shape[0]
        confidence_scores_weak = combined_confidence_scores[:batch_size]
        confidence_scores_strong = combined_confidence_scores[batch_size:]
        
        # 计算交叉熵损失
        loss_zero = torch.nn.CrossEntropyLoss()(confidence_scores_strong, confidence_scores_weak.argmax(dim=1))
        #L2有待实现
        Loss_total = loss_nonzero + alpha * loss_zero
        self.optim.zero_grad()
        Loss_total.backward()
        self.optim.step()

    def update_reward(self, buffer):
        pass

    def update(self, buffer, alpha):
        '''更新reward估计网络，再修改奖励'''
        self.update_network(buffer, alpha)

        
        
    def estimate(self,buffer):
        pass

