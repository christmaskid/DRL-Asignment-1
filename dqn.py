import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import math
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, action_size),
            nn.BatchNorm1d(action_size),
        )

    def forward(self, x):
        # print("input", x.shape, x)
        x = self.layers(x)
        # print("output", x.shape, x)
        return x

class DQNTrainer:
    def __init__(self, state_size, action_size, hidden_size=64, lr=0.1, gamma=0.99, device="cuda", batch_size=16):
        self.dqn = DQN(state_size, action_size, hidden_size).to(device)
        self.target_dqn = DQN(state_size, action_size, hidden_size).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())  # Sync
        self.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = optim.SGD(self.dqn.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer = []

    def get_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(range(self.action_size)))
        else:
            with torch.no_grad():
                q_values = self.dqn(torch.tensor([state], dtype=torch.float32, device=self.device))
                action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) == self.batch_size:
          self.update_batch()

    def update_batch(self):
        states, actions, rewards, next_states, dones = zip(*(self.buffer))
        # states, actions, rewards, next_states = zip(*(self.buffer))
        self.dqn.train()
        # print("Update batch by actions", actions)
        
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # q_table[state][action] += alpha * (reward + gamma * q_table[next_state][action])
        # target = alpha * (reward + gamma * torch.max(q_table[next_state]).detach())
        # print("states", states)
        q_values = self.dqn(states) #.gather(1, actions).squeeze()
        # print("outputs", q_values)
        q_values = q_values.gather(1, actions).squeeze()
        # print("q_values", q_values)
        next_q_values = self.target_dqn(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        self.optimizer.zero_grad()
        loss = self.loss_func(q_values, targets.detach())
        loss.backward()
        self.optimizer.step()
        print(f"\rLoss: {loss.item():.4f}", end='')

        self.buffer.clear()
        self.dqn.eval()

