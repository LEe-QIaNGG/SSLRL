# -*- coding: utf-8 -*-
import torch
import numpy as np
from network import ResNet
    
class Reward_Estimator:
    def __init__(self, obs_dim, act_dim,device):
        '''要求环境的action是discrete
        

        '''
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_reward = 5
        self.Qnet = ResNet(obs_dim+act_dim, self.num_reward).to(device)
        self.Vnet = ResNet(obs_dim, self.num_reward ).to(device)
        self.optim_Q= torch.optim.Adam(self.Qnet.parameters(), lr=1e-3)
        self.optim_V= torch.optim.Adam(self.Vnet.parameters(), lr=1e-3)
        self.reward_list=[-2,-1,0,1,2]
        self.true_reward=[]
        self.threshold=0.7
        self.device=device

    def get_input_data(self, buffer, mask_nonzero):
        obs = torch.tensor(buffer.obs[mask_nonzero], device=self.device)
        action = torch.tensor(buffer.act[mask_nonzero], device=self.device)
        next_obs = torch.tensor(buffer.obs_next[mask_nonzero], device=self.device)
        action = action.unsqueeze(1)  # 添加这一行
        return torch.cat([obs, next_obs, action], dim=-1).float()
    
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
    
    def calculate_mask(self, buffer):
        # 使用numpy的isin函数来创建掩码
        # ~操作符用于取反，因为我们要找的是不在true_reward中的项
        return ~np.isin(buffer.rew, self.true_reward)

    def update_network(self, buffer, alpha):
        # 分别计算非零奖励和零奖励的损失
        mask_nonzero = buffer.rew != 0
        mask_zero = buffer.rew == 0

        # # 计算真实奖励的损失
        input_data_nonzero = self.get_input_data(buffer, mask_nonzero)
        confidence_scores, loss2 = self.get_QVconfidence(input_data_nonzero, is_L2=True)
        max_confidence, max_indices = torch.max(confidence_scores, dim=1)
        predicted_reward = torch.tensor([self.reward_list[i] if conf > self.threshold else 0 for i, conf in zip(max_indices, max_confidence)])
        loss_nonzero = torch.nn.MSELoss()(predicted_reward, torch.tensor(buffer.rew[mask_nonzero]))
        
        #constancy regularization
        input_data_zero = self.get_input_data(buffer, mask_zero)
        input_data_zero_weak = self.weak_augment(input_data_zero)
        input_data_zero_strong = self.strong_augment(input_data_zero)

        confidence_scores_weak,loss_constancy_weak = self.get_QVconfidence(input_data_zero_weak, is_L2=True)
        confidence_scores_strong,loss_constancy_strong = self.get_QVconfidence(input_data_zero_strong, is_L2=True)
        
        # 计算交叉熵损失
        loss_zero = torch.nn.CrossEntropyLoss()(confidence_scores_strong, confidence_scores_weak.argmax(dim=1))

        Loss_total = (1-alpha)*loss_nonzero  + alpha * loss_zero+loss2+loss_constancy_weak+loss_constancy_strong    
        self.optim_Q.zero_grad()
        self.optim_V.zero_grad()
        Loss_total.backward()
        self.optim_Q.step()
        self.optim_V.step()

    def update_reward(self, buffer, alpha):
        # 获取buffer中的obs、obs_next和act
        obs = torch.tensor(buffer.obs)
        obs_next = torch.tensor(buffer.obs_next)
        act = torch.tensor(buffer.act).unsqueeze(-1)
        
        # 拼接输入数据
        input_data = torch.cat([obs, obs_next, act], dim=-1).float().to(self.device)
        
        # 获取当前奖励
        current_rewards = torch.tensor(buffer.rew)
        
        # 对于buffer中reward不等于true_reward里的值的项
        mask = self.calculate_mask(buffer)
        if np.any(mask):
            # 获取满足条件的输入数据
            masked_input = input_data[mask]
                
            # 通过网络获取置信度
            confidence_scores = self.get_QVconfidence(masked_input,is_L2=False)
                
            # 获取最大置信度及其索引
            max_confidence, max_indices = torch.max(confidence_scores.cpu(), dim=1)
                
            # 更新满足条件的奖励
            update_mask = max_confidence > self.threshold
            if torch.any(update_mask):
                new_rewards = torch.tensor([self.reward_list[i] for i in max_indices[update_mask]])
                buffer.rew[mask][update_mask] = new_rewards.numpy()
        

    def update(self, batch, buffer, alpha, iter):
        '''更新reward估计网络，再修改奖励'''
        update_flag = False
        if iter == 1:
            buffer_rew = torch.tensor(buffer.rew)
            if not torch.all(buffer_rew == 0):
                self.update_network(buffer, alpha)
                update_flag = True
        else:
            batch_rew = torch.tensor(batch.rew)
            if not torch.all(batch_rew == 0):
                self.update_network(batch, alpha)
                update_flag = True
        
        if update_flag:
            self.update_reward(buffer,alpha)
            print("奖励不为零")
        # else:
        #     print("奖励全为0，不更新")

        
        
    def get_QVconfidence(self, input_data, is_L2):
        # 分离输入数据
        obs = input_data[:, :self.obs_dim]
        obs_next = input_data[:, self.obs_dim:2*self.obs_dim]
        action = input_data[:, -self.act_dim:]
        
        # 拼接obs和action作为Qnet的输入
        q_input = torch.cat([obs, action], dim=-1)
        Q_confidence = self.Qnet(q_input)
        
        # 使用obs_next作为Vnet的输入
        V_confidence_next = self.Vnet(obs_next)
        total_confidence = (Q_confidence + V_confidence_next)/2

        #计算L2
        if is_L2:
            V_confidence=self.Vnet(obs)
            L2=self.calculate_L2(V_confidence, Q_confidence)
            return total_confidence, L2
        return total_confidence

    def calculate_L2(self, V_confidence, Q_confidence):
        # 获取V和Q的最大置信度值及其索引
        V_max_confidence, V_max_indices = torch.max(V_confidence, dim=1)
        Q_max_confidence, Q_max_indices = torch.max(Q_confidence, dim=1)
        
        # 创建掩码，找出大于阈值的项
        V_mask = V_max_confidence > self.threshold
        Q_mask = Q_max_confidence > self.threshold
        
        # 取V_mask和Q_mask的交集
        combined_mask = V_mask & Q_mask
        
        # 获取满足条件的奖励值
        V_rewards = torch.tensor([self.reward_list[i] for i in V_max_indices[combined_mask]])
        Q_rewards = torch.tensor([self.reward_list[i] for i in Q_max_indices[combined_mask]])
        
        # 只在Q比V大时计算MSE损失
        mask = Q_rewards > V_rewards
        if mask.any():
            L2 = torch.nn.MSELoss()(V_rewards[mask].float(), Q_rewards[mask].float())
        else:
            L2 = torch.tensor(0.0)
        
        return L2

