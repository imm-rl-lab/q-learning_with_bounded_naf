import numpy as np
import torch


class VanDerPol:
    def __init__(self, initial_state=np.array([0, 1, 0]), action_min=np.array([-1]), action_max=np.array([+1]),
                 terminal_time=11, dt=0.1, inner_step_n=10):
        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.beta = 0.05
        self.r = 0.05
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.initial_state = initial_state
        self.state = self.initial_state

    def reset(self):
        self.state = self.initial_state
        return self.state

    def g(self, state):
        return torch.stack([torch.zeros(state.shape[1]), torch.ones(state.shape[1])]).transpose(0, 1).unsqueeze(1).type(torch.FloatTensor)

    def set_dt(self, dt):
        self.dt = dt
        self.inner_dt = dt / self.inner_step_n

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            f = np.array([1, self.state[2], (1 - self.state[1] ** 2) * self.state[2] - self.state[1] + action[0]])
            self.state = self.state + f * self.inner_dt

        if self.state[0] < self.terminal_time:
            done = False
            reward = - self.r * action[0] ** 2 * self.dt
        else:
            done = True
            reward = - self.state[1] ** 2 - self.state[2] ** 2

        return self.state, reward, done, None

    def get_state_obs(self):
        return 'time: %.3f  x: %.3f y: %.3f' % (self.state[0], self.state[1], self.state[2])

    def render(self):
        print(self.get_state_obs())
