import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

#  model 1
class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

#  model 2
# class DeepQNet(nn.Module):
#     def __init__(self, input_size=12, hidden_size1=128, hidden_size2=64, output_size=4):
#         super().__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size1)
#         self.linear2 = nn.Linear(hidden_size1, hidden_size2)
#         self.linear3 = nn.Linear(hidden_size2, output_size)

#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x

#  model 3
# class Conv_QNet(nn.Module):
#     def __init__(self, input_channels, output_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 20 * 20, 256)
#         self.fc2 = nn.Linear(256, output_size)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
class Training:
    def __init__(self, q_table, lr, discount_factor):
        self.lr = lr
        self.discount_factor = discount_factor
        self.q_table = q_table
        self.optimizer = optim.Adam(q_table.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_in_iteration(self, state, action, next_state, reward):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        pred = self.q_table(state)

        with torch.no_grad():
            max_next_q = torch.max(self.q_table(next_state))
        target = pred.clone()
        target[action] = ((1 - self.lr) * target[action]) + self.lr * (reward + self.discount_factor * max_next_q)
      
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()