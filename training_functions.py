import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class reward_estimator:
    def __init__(self):
        self.net = Net()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)
    def update(self, buffer, epoch):
        # 根据epoch选择reward为0或非0的记录
        if epoch % 2 == 0:
            # 偶数epoch选择非零reward记录
            mask = buffer.rew != 0
        else:
            # 奇数epoch选择零reward记录
            mask = buffer.rew == 0
        
        obs = buffer.obs[mask]
        action = buffer.act[mask]
        next_obs = buffer.obs_next[mask]
        reward = buffer.rew[mask]

        input_data = torch.cat([obs, action, next_obs], dim=-1)
        predicted_reward = self.net(input_data)

        loss = torch.nn.MSELoss()(predicted_reward, reward)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
    def estimate(self,buffer):
        pass

def train_fn(epoch: int, env_step: int, args,logger,policy) -> None:
    # nature DQN setting, linear decay in the first 1M steps
    if env_step <= 1e6:
        eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
    else:
        eps = args.eps_train_final
    policy.set_eps(eps)
    if env_step % 1000 == 0:
        logger.write("train/env_step", env_step, {"train/eps": eps})
    #第一个epoch初始化reward estimator，update estimator网络，更新buffer的reward值
    print(train_collector.buffer)