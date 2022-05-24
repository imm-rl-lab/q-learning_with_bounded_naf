import numpy as np
import torch


class DubinsCar:
    def __init__(self, initial_state=np.array([0, 0, 0, 0]), dt=0.1, terminal_time=2 * np.pi, inner_step_n=20,
                 action_min=np.array([-1]), action_max=np.array([1])):
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.beta = 0.01
        self.r = 0.01
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n

    def reset(self):
        self.state = self.initial_state
        return self.state

    def set_dt(self, dt):
        self.dt = dt
        self.inner_dt = dt / self.inner_step_n

    def g(self, state):
        return torch.stack([torch.zeros(state.shape[1]), 
                            torch.zeros(state.shape[1]), 
                            torch.ones(state.shape[1]) * 0.75]).transpose(0, 1).unsqueeze(1).type(torch.FloatTensor)

    def step(self, action):
        action_raw = action.copy()
        action = np.clip(action, self.action_min, self.action_max)
        action = action * 0.75 + 0.25

        for _ in range(self.inner_step_n):
            self.state = self.state + np.array(
                [1, np.cos(self.state[3]), np.sin(self.state[3]), action[0]]) * self.inner_dt

        if self.state[0] >= self.terminal_time:
            reward = -np.abs(self.state[1] - 4) - np.abs(self.state[2]) - np.abs(self.state[3] - 0.75 * np.pi)
            done = True
        else:
            reward = -self.r * (action_raw[0] ** 2) * self.dt
            done = False

        return self.state, reward, done, None

    def get_state_obs(self):
        return 'time: %.3f  x: %.3f y: %.3f theta: %.3f' % (self.state[0], self.state[1], self.state[2], self.state[3])

    def render(self):
        print(self.get_state_obs())
