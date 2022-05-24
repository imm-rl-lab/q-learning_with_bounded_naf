import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

from models.linear_transformations import transform_interval
from models.q_models import MuModel


class DDPG:

    def __init__(self, action_min, action_max, q_model, pi_model, noise,
                 q_model_lr=1e-3, pi_model_lr=1e-4, gamma=0.99, batch_size=64, tau=1e-3,
                 memory_len=6000000, learning_iter_per_fit=1, convex_comb_for_actions=False, clip=False):

        self.action_min = torch.FloatTensor(action_min)
        self.action_max = torch.FloatTensor(action_max)
        self.q_model = q_model
        self.clip = clip
        self.pi_model = MuModel(pi_model, clip)
        self.noise = noise

        self.q_model_lr = q_model_lr
        self.pi_model_lr = pi_model_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory = deque(maxlen=memory_len)
        self.learning_iter_per_fit = learning_iter_per_fit
        self.convex_comb_for_actions = convex_comb_for_actions

        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_model_lr)
        self.q_target_model = deepcopy(self.q_model)
        self.pi_target_model = deepcopy(self.pi_model)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        if self.convex_comb_for_actions:
            action = (1 - self.noise.threshold) * self.pi_model(state) + torch.FloatTensor(self.noise.noise())
        else:
            action = self.pi_model(state) + torch.FloatTensor(self.noise.noise())
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action.detach().numpy(), self.action_min.numpy(), self.action_max.numpy())

    def train(self):
        self.noise.threshold = 1
        self.memory.clear()
        self.q_model.train()

    def eval(self):
        self.noise.threshold = 0
        self.q_model.eval()

    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        return None

    def add_to_memory(self, step):
        self.memory.append(step)

    def fit(self, step):
        self.add_to_memory(step)

        if len(self.memory) >= self.batch_size:
            for _ in range(self.learning_iter_per_fit):
            
                batch = random.sample(self.memory, self.batch_size)
                states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
                rewards = rewards.reshape(self.batch_size, 1)
                dones = dones.reshape(self.batch_size, 1)

                pred_next_actions = transform_interval(self.pi_target_model(next_states),
                                                       self.action_min, self.action_max)
                next_states_and_pred_next_actions = torch.cat((next_states, pred_next_actions), dim=1)
                targets = rewards + (1 - dones) * self.gamma * self.q_target_model(next_states_and_pred_next_actions)
                states_and_actions = torch.cat((states, actions), dim=1)
                q_loss = torch.mean((self.q_model(states_and_actions) - targets.detach()) ** 2)
                self.update_target_model(self.q_target_model, self.q_model, self.q_optimizer, q_loss)

                pred_actions = transform_interval(self.pi_model(states), 
                                                  self.action_min, self.action_max)
                states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
                pi_loss = - torch.mean(self.q_model(states_and_pred_actions))
                self.update_target_model(self.pi_target_model, self.pi_model, self.pi_optimizer, pi_loss)
                
        return None