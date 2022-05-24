import numpy as np


def load_noise(state_dict):
    action_dimension = state_dict['action_dim']
    mu = state_dict['mu']
    theta = state_dict['theta']
    sigma = state_dict['sigma']
    threshold_min = state_dict['threshold_min']
    threshold_decrease = state_dict['threshold_decrease']
    return OUNoise(action_dimension, mu, theta, sigma, threshold_min, threshold_decrease)


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, threshold_min=0.01,
                 threshold_decrease=1e-3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.threshold = 1
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.threshold

    def decrease(self):
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease

    def state_dict(self):
        return {
            'action_dim': self.action_dimension,
            'mu': self.mu,
            'theta': self.theta,
            'sigma': self.sigma,
            'threshold_min': self.threshold_min,
            'threshold_decrease': self.threshold_decrease
        }
