import random
import numpy as np
import torch
from collections import deque
from model import DQN, Trainer
from memory import ReplayMemory

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.state_memory = deque(maxlen=3)

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.gamma = 0.95

        # Model + Trainer
        self.model = DQN(state_size, action_size)
        self.trainer = Trainer(self.model, lr=0.001, gamma=self.gamma)

        # Replay memory
        self.memory = ReplayMemory(5000)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state)
        return torch.argmax(prediction).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self, batch_size=32):
        if self.memory.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_stacked_state(self, state):
        self.state_memory.append(state)

        # Fill initial memory
        while len(self.state_memory) < 3:
            self.state_memory.append(state)

        return np.concatenate(self.state_memory)
    
    def reset_memory(self):
        self.state_memory.clear()