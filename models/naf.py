import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from copy import deepcopy
from models.linear_transformations import transform_interval


class NAF:
    def __init__(self, action_min, action_max, q_model, noise,
                 batch_size=128, gamma=1, tau=1e-2, q_model_lr=1e-3, memory_size=10000000):
        self.action_min = action_min
        self.action_max = action_max
        self.q_model = q_model
        self.noise = noise
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = q_model_lr
        
        self.opt = torch.optim.Adam(self.q_model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.q_target = deepcopy(self.q_model)
        self.memory = deque(maxlen=memory_size)
        return None

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

    def get_action(self, state):
        state = torch.FloatTensor(state)
        state.requires_grad = True
        mu_value = self.q_model.mu_model(state).detach().numpy()
        noise = self.noise.noise()
        action = mu_value + noise
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action, self.action_min, self.action_max)

    def update_targets(self, target, original,loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * original_param.data)
        return None

    def fit(self, step):
        self.memory.append(step)

        if len(self.memory) >= self.batch_size:
            
            #get batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            states.requires_grad = True
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            #get loss
            target = rewards + (1 - dones) * self.gamma * self.q_target.v_model(next_states).detach()
            q_values = self.q_model(states, actions)
            loss = self.loss(q_values, target)
            
            #train
            self.update_targets(self.q_target, self.q_model, loss)
        
        return None
