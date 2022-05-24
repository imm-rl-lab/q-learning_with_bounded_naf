import time
import numpy as np


class Solver:
    def __init__(self, env):
        self.env = env
        self.total_rewards = []
        self.mean_total_rewards = []
        return None

    def __callback__(self, agent, epoch, total_reward):
        self.total_rewards.append(total_reward)
        mean_total_reward = np.mean(self.total_rewards[max(0, epoch - 25):epoch + 1])
        self.mean_total_rewards.append(mean_total_reward)
        print("epoch=%.0f, noise threshold=%.3f, total reward=%.3f, mean reward=%.3f, " % (
            epoch, agent.noise.threshold, total_reward, mean_total_reward) + self.env.get_state_obs())
        return None

    def _evaluate_(self, agent, agent_learning=False, render=False):
        agent.noise.reset()
        state = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            if agent_learning:
                agent.fit([state, action, reward, done, next_state])
            
            if render:
                self.env.render()

            state = next_state
            total_reward += reward

        if agent_learning:
            agent.noise.decrease()

        return total_reward

    def train(self, agent, learning_config):
        epoch_num = learning_config['epoch_num']
        render = learning_config.get('render', False)
        
        for epoch in range(epoch_num):
            rewards = self._evaluate_(agent, agent_learning=True, render=render)
            total_reward = np.sum(rewards)
            self.__callback__(agent, epoch, total_reward)
            
        print('Training finished!')
        return self.mean_total_rewards

    def evaluate(self, agent):
        agent.noise.threshold = 0
        total_reward = self._evaluate_(agent, agent_learning=False, render=True)
        print('Evaluation finished!')
        print('Final state: ')
        print(self.env.state)
        print('Total Reward: ')
        print(total_reward)
        return None
