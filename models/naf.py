import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from models.linear_transformations import transform_interval


class NAF:
    def __init__(self, action_min, action_max, q_model, noise,
                 batch_size=64, gamma=1, tau=1e-2, q_model_lr=1e-3):
        self.action_max = action_max
        self.action_min = action_min
        self.q_model = q_model
        self.opt = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        self.loss = nn.MSELoss()
        self.lr = q_model_lr
        self.q_target = deepcopy(self.q_model)
        self.tau = tau
        self.memory = deque(maxlen=1000000)
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise = noise
        self.learning_n_per_fit = 1

    def save(self, path):
        torch.save({
            'q-model': self.q_model.state_dict(),
            'noise': self.noise.state_dict(),
            'action_min': self.action_min,
            'action_max': self.action_max,
            'tau': self.tau,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }, path)

    def train(self):
        self.noise.threshold = 1
        self.memory.clear()
        self.q_model.train()

    def eval(self):
        self.noise.threshold = 0
        self.q_model.eval()

    def get_action(self, state):
        state = torch.FloatTensor(state)
        state.requires_grad = True
        mu_value = self.q_model.mu_model(state).detach().numpy()
        noise = self.noise.noise()
        action = mu_value + noise
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action, self.action_min, self.action_max)

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * original_param.data)

    def add_to_memory(self, step):
        self.memory.append(step)

    def fit(self, step):
        self.add_to_memory(step)

        if len(self.memory) >= self.batch_size:
            batch = list(zip(*random.sample(self.memory, self.batch_size)))
            states = torch.FloatTensor(np.array(batch[0]))
            actions = torch.FloatTensor(np.array(batch[1]))
            rewards = torch.FloatTensor(np.array(batch[2]))
            dones = torch.FloatTensor(np.array(batch[3]))
            next_states = torch.FloatTensor(np.array(batch[4]))
            states.requires_grad = True
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            target = rewards + (1 - dones) * self.gamma * self.q_target.v_model(next_states).detach()
            q_values = self.q_model(states, actions)
            loss = self.loss(q_values, target)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.update_targets(self.q_target, self.q_model)
