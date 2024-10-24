# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision
from network import ResNet,FCNet
    
class Reward_Estimator:
    def __init__(self, obs_dim, act_dim,device,network_type='FCNet',data_augmentation=None,is_L2=False):
        '''要求环境的action是discrete
        

        '''
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_reward = 10
        if network_type == 'ResNet':
            self.Qnet = ResNet(obs_dim+act_dim, self.num_reward).to(device)
            self.Vnet = ResNet(obs_dim, self.num_reward ).to(device)
        elif network_type == 'FCNet':
            self.Qnet = FCNet(obs_dim+act_dim, self.num_reward).to(device)
            self.Vnet = FCNet(obs_dim, self.num_reward).to(device)
        self.optim_Q= torch.optim.Adam(self.Qnet.parameters(), lr=1e-3)
        self.optim_V= torch.optim.Adam(self.Vnet.parameters(), lr=1e-3)
        self.reward_list = [0] * self.num_reward
        self.true_reward=[]
        self.threshold=0.7
        self.device=device
        self.data_augmentation=data_augmentation
        self.is_L2=is_L2

    def get_input_data(self, buffer, mask_nonzero):
        obs = torch.tensor(buffer.obs[mask_nonzero], device=self.device)
        action = torch.tensor(buffer.act[mask_nonzero], device=self.device)
        next_obs = torch.tensor(buffer.obs_next[mask_nonzero], device=self.device)
        action = action.unsqueeze(1)  # 添加这一行
        return torch.cat([obs, next_obs, action], dim=-1).float()
    
    def shannon_augment(self, input_data, n=16):
        data_without_action = input_data[:, :-self.act_dim]
        action = input_data[:, -self.act_dim:]
        
        # 将data_without_action纵向分为n个块
        chunk_size = data_without_action.shape[1] // n
        chunks = [data_without_action[:, i*chunk_size:(i+1)*chunk_size] for i in range(n)]
        
        # 计算每个块的香农熵并乘以相应的块
        augmented_chunks = []
        for chunk in chunks:
            # 将数据离散化
            binned_data = torch.histc(chunk, bins=256, min=chunk.min(), max=chunk.max())
            # 计算概率分布
            probs = binned_data / binned_data.sum()
            # 计算香农熵
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
            # 将块乘以其香农熵
            augmented_chunks.append(chunk * entropy.item())
        
        # 拼接增强后的数据块
        augmented_data = torch.cat(augmented_chunks, dim=1)
        
        # 重新组合数据并返回
        return torch.cat([augmented_data, action], dim=-1) 

    def cutout_augment(self, input_data):
        n = int(np.log2(self.act_dim) / 2)
        # 分离action列
        data_without_action = input_data[:, :-self.act_dim]
        action = input_data[:, -self.act_dim:]
        
        # 随机选择n列
        num_cols = data_without_action.shape[1]
        cols_to_zero = torch.randperm(num_cols)[:n]
        
        # 将选中的列置为零
        data_without_action[:, cols_to_zero] = 0
        
        # 重新组合数据
        return torch.cat([data_without_action, action], dim=-1)

    def GaussianNoise_augment(self, input_data,sigma=0.1):
        # 分离action列
        data_without_action = input_data[:, :-self.act_dim]
        action = input_data[:, -self.act_dim:]
        
        # 对除action外的数据添加高斯噪声
        noise = torch.randn_like(data_without_action) * sigma  # 0.1是噪声强度，可以根据需要调整
        augmented_data = data_without_action + noise
        
        # 重新组合数据
        return torch.cat([augmented_data, action], dim=-1)
    
    def flip_augment(self, input_data):
        # 分离action列
        data_without_action = input_data[:, :-self.act_dim]
        action = input_data[:, -self.act_dim:]
        
        # 将data_without_action分为obs和obs_next
        obs_dim = data_without_action.shape[1] // 2
        obs = data_without_action[:, :obs_dim]
        obs_next = data_without_action[:, obs_dim:]
        
        # 分别对obs和obs_next进行翻转操作
        flipped_obs = torch.flip(obs, dims=[1])
        flipped_obs_next = torch.flip(obs_next, dims=[1])
        
        # 重新组合翻转后的数据
        flipped_data = torch.cat([flipped_obs, flipped_obs_next], dim=1)     
        # 重新组合数据
        return torch.cat([flipped_data, action], dim=-1)
    
    def scale_augment(self, input_data, scale_range=(0.8, 1.2)):
        # 分离action列
        data_without_action = input_data[:, :-self.act_dim]
        action = input_data[:, -self.act_dim:]
        
        # 将data_without_action分为obs和obs_next
        obs_dim = data_without_action.shape[1] // 2
        obs = data_without_action[:, :obs_dim]
        obs_next = data_without_action[:, obs_dim:]
        
        # 为每个样本生成随机缩放因子
        batch_size = obs.shape[0]
        scale_factors = torch.empty(batch_size, 1).uniform_(scale_range[0], scale_range[1]).to(obs.device)
        
        # 对obs和obs_next分别进行缩放
        scaled_obs = obs * scale_factors
        scaled_obs_next = obs_next * scale_factors
        
        # 重新组合缩放后的数据
        scaled_data = torch.cat([scaled_obs, scaled_obs_next], dim=1)
        
        # 重新组合数据并返回
        return torch.cat([scaled_data, action], dim=-1)
    
    def translate_augment(self, input_data, translate_range=(0.1, 0.1)):
        # 分离action列
        data_without_action = input_data[:, :-self.act_dim]
        action = input_data[:, -self.act_dim:]
        
        # 将data_without_action分为obs和obs_next
        obs_dim = data_without_action.shape[1] // 2
        obs = data_without_action[:, :obs_dim]
        obs_next = data_without_action[:, obs_dim:]
        
        # 创建RandomAffine实例用于平移
        transform = torchvision.transforms.RandomAffine(
            degrees=0,  # 不进行旋转
            translate=translate_range  # 平移范围
        )
        
        # 分别对obs和obs_next进行平移操作
        translated_obs = transform(obs.unsqueeze(0)).squeeze(0)
        translated_obs_next = transform(obs_next.unsqueeze(0)).squeeze(0)
        
        # 重新组合平移后的数据
        translated_data = torch.cat([translated_obs, translated_obs_next], dim=1)
        
        # 重新组合数据并返回
        return torch.cat([translated_data, action], dim=-1)
        
    
    def smooth_augment(self, input_data, n=3):
        # 分离action列
        data_without_action = input_data[:, :-self.act_dim]
        action = input_data[:, -self.act_dim:]
        
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
        is_L2=self.is_L2
        # 分别计算非零奖励和零奖励的损失
        mask_nonzero = buffer.rew != 0
        mask_zero = buffer.rew == 0

        # # 计算真实奖励的损失
        input_data_nonzero = self.get_input_data(buffer, mask_nonzero)
        confidence_scores, loss2 = self.get_QVconfidence(input_data_nonzero, is_L2=is_L2)
        max_confidence, max_indices = torch.max(confidence_scores, dim=1)
        predicted_reward = torch.tensor([self.reward_list[i] if conf > self.threshold else 0 for i, conf in zip(max_indices, max_confidence)])
        loss_nonzero = torch.nn.MSELoss()(predicted_reward, torch.tensor(buffer.rew[mask_nonzero]))
        
        #constancy regularization
        input_data_zero = self.get_input_data(buffer, mask_zero)

        #data augmentation
        input_data_zero_weak = self.GaussianNoise_augment(input_data_zero)
        if self.data_augmentation=='shannon':
            input_data_zero_strong = self.shannon_augment(input_data_zero)
        elif self.data_augmentation=='cutout':
            input_data_zero_strong = self.cutout_augment(input_data_zero)
        elif self.data_augmentation=='smooth':
            input_data_zero_strong = self.smooth_augment(input_data_zero)
        elif self.data_augmentation=='scale':
            input_data_zero_strong = self.scale_augment(input_data_zero)
        elif self.data_augmentation=='translate':
            input_data_zero_strong = self.translate_augment(input_data_zero)
        elif self.data_augmentation=='flip':
            input_data_zero_strong = self.flip_augment(input_data_zero)

        confidence_scores_weak,loss_constancy_weak = self.get_QVconfidence(input_data_zero_weak, is_L2=is_L2)
        confidence_scores_strong,loss_constancy_strong = self.get_QVconfidence(input_data_zero_strong, is_L2=is_L2)
        # 计算交叉熵损失
        loss_zero = torch.nn.CrossEntropyLoss()(confidence_scores_strong, confidence_scores_weak.argmax(dim=1))
        if is_L2:
            loss2=loss_constancy_weak+loss_constancy_strong+loss2
            Loss_total = (1-alpha)*loss_nonzero  + alpha * loss_zero + loss2    
        else:
            Loss_total = (1-alpha)*loss_nonzero  + alpha * loss_zero
        self.optim_Q.zero_grad()
        self.optim_V.zero_grad()
        Loss_total.backward()
        self.optim_Q.step()
        self.optim_V.step()

    def update_true_reward(self, reward):
        non_zero_rewards = reward[reward != 0]
        if len(non_zero_rewards) == 0:
            return
        old_min = min(self.true_reward) if self.true_reward else float('inf')
        old_max = max(self.true_reward) if self.true_reward else float('-inf')
        
        for r in non_zero_rewards:
            if r.item() not in self.true_reward:
                self.true_reward.append(r.item())
        
        new_min = min(self.true_reward)
        new_max = max(self.true_reward)
        
        if new_min != old_min or new_max != old_max:
            min_value = min(new_min, 0)
            self.reward_list = np.linspace(min_value, new_max-1, self.num_reward).tolist()
            print('\nreward list:', self.reward_list)
            print('\ntrue reward:', self.true_reward)

    def update_reward(self, buffer,iter,alpha,num_iter=200000):
        # 获取buffer中的obs、obs_next和act
        obs = torch.tensor(buffer.obs)
        obs_next = torch.tensor(buffer.obs_next)
        act = torch.tensor(buffer.act).unsqueeze(-1)
        input_data = torch.cat([obs, obs_next, act], dim=-1).float().to(self.device)
        
        # 对于buffer中reward不等于true_reward里的值的项
        mask = self.calculate_mask(buffer)
        # if iter%100 == 0:
        #     print('真实reward的数量:', np.sum(~mask))
        num_real_reward=np.sum(~mask)
        mask = torch.from_numpy(mask)
        if iter<num_iter/3:
            update_prob=np.log(num_real_reward/len(mask))
        elif iter<2*num_iter/3:
            update_prob=num_real_reward/len(mask)
        else:
            update_prob=1-alpha
        mask = torch.where(torch.rand_like(mask.float()) < update_prob, torch.zeros_like(mask,dtype=torch.bool), mask)

        # if iter%100 == 0:
        #     print("\nmask:",mask)  
        #     print('len:',len(mask))
        #     print('更新数量:', torch.sum(mask))
        if torch.any(mask):
            # 获取满足条件的输入数据
            masked_input = input_data[mask]
                
            # 通过网络获取置信度
            confidence_scores, _ = self.get_QVconfidence(masked_input,is_L2=False)
            # 获取最大置信度及其索引
            max_confidence, max_indices = torch.max(confidence_scores.cpu(), dim=1)
                
            # 更新满足条件的奖励
            update_mask = max_confidence > self.threshold
            if torch.any(update_mask):
                new_rewards = torch.tensor([self.reward_list[i] for i in max_indices[update_mask]])
                buffer.rew[mask][update_mask] = new_rewards.numpy()
        
    def update(self, batch, buffer, alpha, iter):
        update_flag = False
        if iter == 1:
            buffer_rew = torch.tensor(buffer.rew)
            self.update_true_reward(buffer_rew)
            if not torch.all(buffer_rew == 0):
                self.update_network(buffer, alpha)
                update_flag = True
        else:
            batch_rew = torch.tensor(batch.rew)
            self.update_true_reward(batch_rew)
            if not torch.all(batch_rew == 0):
                self.update_network(batch, alpha)
                update_flag = True
        
        if update_flag:
            self.update_reward(buffer,iter,alpha)


        
        
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
        return total_confidence, torch.tensor(0.0)

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


