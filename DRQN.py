import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F


class DRQN(nn.Module):
    def __init__(self, action_size, sequence_length, img_size):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(64 * ((img_size // 4) // 2)
                            ** 2, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        batch_size, sequence_length, img_size, _, _ = x.size()
        x = F.relu(self.conv1(x.view(-1, img_size, img_size, 1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x, _ = self.lstm(x.view(batch_size, sequence_length, -1))
        x = F.relu(self.fc1(x[:, -1, :]))
        return self.fc2(x)


class Agent:
    def __init__(self, env, batch_size=64, max_experiences=5000):
        self.env = env
        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.max_experiences = max_experiences
        self.memory = deque(maxlen=self.max_experiences)
        self.batch_size = batch_size

        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.episodes = 1000

        self.network = DRQN()
        self.target_network = DRQN()
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.RMSprop(
            self.parameters(), lr=0.00025, eps=self.epsilon_min)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                pred = self.network(state)
                return pred.argmax().item()

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.update_epsilon()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def play(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.add_experience(state, action, reward, next_state, done)
            state = next_state

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(
                next_state, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.long)

            current_q_values = self(state)[0][action]
            next_q_values = self.target_network(next_state).max(1)[0]
            target_q_values = reward + self.gamma * next_q_values * (1 - done)

            loss = self.loss_fn(current_q_values, target_q_values.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update target network
        self.target_network.load_state_dict(self.state_dict())

    def train(self):
        for _ in range(self.episodes):
            self.play()
            self.learn()
