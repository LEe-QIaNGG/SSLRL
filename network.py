
import torch
import numpy as np

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, in_features)
        self.fc2 = torch.nn.Linear(in_features, in_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        return self.relu(out)

class ResNet(torch.nn.Module):
    def __init__(self, input_dim, num_reward, num_blocks=3):
        super(ResNet, self).__init__()
        self.input_dim = input_dim
        self.fc_in = torch.nn.Linear(self.input_dim, 64)
        self.blocks = torch.nn.ModuleList([ResidualBlock(64) for _ in range(num_blocks)])
        self.fc_out = torch.nn.Linear(64, num_reward)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x

class FCNet(torch.nn.Module):
    def __init__(self, input_dim, num_reward):
        super(FCNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc_out = torch.nn.Linear(32, num_reward)
        self.dropout = torch.nn.Dropout(0.2)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x