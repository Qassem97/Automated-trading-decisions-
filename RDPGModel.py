import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from torch.utils.tensorboard import SummaryWriter


class RDPGModel(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim):
        super(RDPGModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state):
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        q_values = self.fc(lstm_out[:, -1, :])  # Using only the last output
        return q_values, hidden_state


class RDPGTrainer:
    def __init__(self, env, model, target_model, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.writer = SummaryWriter()

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_experience(self):
        indices = random.sample(range(len(self.buffer)), k=self.batch_size)
        batch = [self.buffer[idx] for idx in indices]
        return zip(*batch)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, episodes=10):
        for episode in range(episodes):
            state = self.env.reset()
            hidden_state = (torch.zeros(1, 1, 50), torch.zeros(1, 1, 50))
            episode_reward = 0
            while not self.env.done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                action_values, hidden_state = self.model(state_tensor, hidden_state)
                action = self.select_action(action_values)
                next_state, reward, done, _ = self.env.step(action)

                self.add_experience((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                if done:
                    break

            self.update_target_model()
            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            return torch.argmax(q_values).item()  # Best action

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return  # Ensure there's enough data to sample from

        states, actions, rewards, next_states, dones = self.sample_experience()
        states = torch.FloatTensor(np.array(states)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        q_values, _ = self.model(states, None)
        next_q_values, _ = self.target_model(next_states, None)

        # Calculate target Q values
        target_q_values = rewards + (self.gamma * next_q_values.max(dim=1)[0].unsqueeze(1) * (1 - dones))

        # Gather the Q-values of the executed actions
        current_q_values = q_values.gather(1, actions)

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


