# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision
from network import ResNet,FCNet
import os   
from data_augmentation import (
    shannon_augment, cutout_augment, gaussian_noise_augment,
    flip_augment, scale_augment, translate_augment, smooth_augment
)
    
class Reward_Estimator:
    def __init__(self,args, act_dim=1,network_type='FCNet'):
        '''要求环境的action是discrete,reward是discrete
        '''
        self.obs_dim = args.state_shape[0]
        self.act_dim = act_dim
        self.num_reward = 12
        if network_type == 'ResNet':
            self.Qnet = ResNet(self.obs_dim+self.act_dim, self.num_reward).to(args.device)
            self.Vnet = ResNet(self.obs_dim, self.num_reward ).to(args.device)
        elif network_type == 'FCNet':
            self.Qnet = FCNet(self.obs_dim+self.act_dim, self.num_reward).to(args.device)
            self.Vnet = FCNet(self.obs_dim, self.num_reward).to(args.device)
        self.optim_Q= torch.optim.Adam(self.Qnet.parameters(), lr=1e-3)
        self.optim_V= torch.optim.Adam(self.Vnet.parameters(), lr=1e-3)
        self.reward_list = [0] * self.num_reward
        self.true_reward=[0]
        self.threshold=0.7
        self.device=args.device
        self.data_augmentation=args.data_augmentation
        self.is_L2=args.is_L2
        self.is_store=args.is_store
        self.task=args.task

    def get_input_data(self, buffer, mask_nonzero):
        obs = torch.tensor(buffer.obs[mask_nonzero], device=self.device)
        action = torch.tensor(buffer.act[mask_nonzero], device=self.device)
        next_obs = torch.tensor(buffer.obs_next[mask_nonzero], device=self.device)
        action = action.unsqueeze(1)  # 添加这一行
        return torch.cat([obs, next_obs, action], dim=-1).float()
    
    def calculate_mask(self, buffer):
        reward_list=np.array(self.true_reward[self.true_reward != 0])
        return ~np.isin(buffer.rew, reward_list)

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
        input_data_zero_weak = gaussian_noise_augment(input_data_zero, self.act_dim)
        if self.data_augmentation == 'shannon':
            input_data_zero_strong = shannon_augment(input_data_zero, self.act_dim)
        elif self.data_augmentation == 'cutout':
            input_data_zero_strong = cutout_augment(input_data_zero, self.act_dim, self.obs_dim)
        elif self.data_augmentation == 'smooth':
            input_data_zero_strong = smooth_augment(input_data_zero, self.act_dim)
        elif self.data_augmentation == 'scale':
            input_data_zero_strong = scale_augment(input_data_zero, self.act_dim)
        elif self.data_augmentation == 'translate':
            input_data_zero_strong = translate_augment(input_data_zero, self.act_dim)
        elif self.data_augmentation == 'flip':
            input_data_zero_strong = flip_augment(input_data_zero, self.act_dim)

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
        if len(non_zero_rewards) == 0 or len(self.true_reward) >= self.num_reward:
            return
        update_flag=False
        for r in non_zero_rewards:
            if r.item() not in self.true_reward:
                self.true_reward.append(r.item())
                update_flag=True
        
        if update_flag:
            if len(self.true_reward) > 0:
                x = np.arange(len(self.true_reward))
                y = np.array(self.true_reward)
                f = np.interp(np.linspace(0, len(self.true_reward) - 1, self.num_reward), x, y)
                self.true_reward = sorted(self.true_reward)
                f = np.sort(f)
                self.reward_list = f.tolist()
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
        if num_real_reward<50:
            update_prob=0.01
        else:
            if iter<num_iter/3:
                update_prob=min(10*num_real_reward/len(mask),0.1)
            else:
                update_prob=min(8*num_real_reward/len(mask),0.1)
        mask = torch.where(torch.rand_like(mask.float()) < update_prob, mask, torch.zeros_like(mask,dtype=torch.bool))
        #mask buffer_size

        if torch.any(mask):
            # 获取满足条件的输入数据
            masked_input = input_data[mask]
                
            # 通过网络获取置信度
            confidence_scores, _ = self.get_QVconfidence(masked_input,is_L2=False)
            # 获取最大置信度及其索引
            max_confidence, max_indices = torch.max(confidence_scores.cpu(), dim=1)
                
            # 更新满足条件的奖励
            update_mask = max_confidence > self.threshold
            if sum(update_mask)>1:
                new_rewards = torch.tensor([self.reward_list[i] for i in max_indices[update_mask]])
                buffer.rew[mask][update_mask] = new_rewards.numpy()

                # if iter%40000>30000 and self.is_store:
                #     reward_log_path = os.path.join("log", "reward_distribution",self.task,str(self.is_L2))
                #     os.makedirs(reward_log_path, exist_ok=True)
                #     n=(iter//40000)+1
                #     rewards_file = os.path.join(reward_log_path, f"rewards_iter_{n}.npy")
                #     if not os.path.exists(rewards_file):
                #         mask_file = os.path.join(reward_log_path, f"mask_iter_{n}.npy")
                #         update_mask_file = os.path.join(reward_log_path, f"update_mask_iter_{n}.npy")
                #         new_rewards_file = os.path.join(reward_log_path, f"new_rewards_iter_{n}.npy")
                #         np.save(rewards_file, buffer.rew)
                #         np.save(mask_file, mask)
                #         np.save(update_mask_file, update_mask)
                #         np.save(new_rewards_file, new_rewards.numpy())
                if iter>190000 and self.is_store:
                    buffer_log_path = os.path.join("log", "buffer",self.task,str(self.is_L2))
                    os.makedirs(buffer_log_path, exist_ok=True)
                    obs_file = os.path.join(buffer_log_path, f"obs.npy")
                    if not os.path.exists(obs_file):    
                        action_file = os.path.join(buffer_log_path, f"action.npy")
                        obs_next_file = os.path.join(buffer_log_path, f"obs_next.npy")
                        rew_file = os.path.join(buffer_log_path, f"rew.npy")
                        mask_file = os.path.join(buffer_log_path, f"mask.npy")
                        update_mask_file = os.path.join(buffer_log_path, f"update_mask.npy")
                        new_rewards_file = os.path.join(buffer_log_path, f"new_rewards.npy")
                        np.save(obs_file, buffer.obs)
                        np.save(action_file, buffer.act)
                        np.save(obs_next_file, buffer.obs_next)
                        np.save(rew_file, buffer.rew)
                        np.save(mask_file, mask)
                        np.save(update_mask_file, update_mask)  
                        np.save(new_rewards_file, new_rewards.numpy())
                    


        
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


