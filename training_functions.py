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
        pass

    def update_network(self, buffer):
        '''更新reward估计网络，再修改奖励'''

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

        
    def estimate(self,buffer):
        pass

