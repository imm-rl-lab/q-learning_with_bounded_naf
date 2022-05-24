import time

import numpy as np
import time


class SingleAgentEvaluationModule:
    def __init__(self, env):
        self.env = env
        self.rewards = np.zeros(0)
        self.mean_rewards = np.zeros(0)

    def __callback__(self, agent, epoch, total_reward):
        self.rewards[epoch] = total_reward
        mean_reward = np.mean(self.rewards[max(0, epoch - 25):epoch + 1])
        self.mean_rewards[epoch] = mean_reward
        print("epoch=%.0f, noise threshold=%.3f, total reward=%.3f, mean reward=%.3f, " % (
            epoch, agent.noise.threshold, total_reward, mean_reward) + self.env.get_state_obs())

    def __reset__(self, epoch_num):
        self.rewards = np.zeros(epoch_num)
        self.mean_rewards = np.zeros(epoch_num)

    def _evaluate_(self, agent, agent_learning=False, render=False):
        total_reward = 0
        state = self.env.reset()
        done = False
        agent.noise.reset()
        while not done:
            if render:
                self.env.render()
            # time.sleep(0.05)
            action = agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            if agent_learning:
                agent.fit([state, action, reward, done, next_state])
            state = next_state
            total_reward += reward
        if agent_learning:
            agent.noise.decrease()
        return total_reward

    def train_agent(self, agent, train_settings):
        epoch_num = train_settings['epoch_num']
        dt_array = train_settings.get('dt', [self.env.dt])
        render = train_settings.get('render', False)
        self.__reset__(epoch_num * len(dt_array))
        dt_idx = 0
        for dt in dt_array:
            self.env.dt = dt
            agent.dt = dt
            agent.train()
            for epoch in range(epoch_num):
                rewards = self._evaluate_(agent, agent_learning=True, render=render)
                total_reward = np.sum(rewards)
                self.__callback__(agent, (dt_idx * epoch_num) + epoch, total_reward)
            dt_idx += 1

        return self.mean_rewards

    def eval_agent(self, agent):
        agent.eval()
        total_reward = self._evaluate_(agent, agent_learning=False, render=True)
        print('Evaluation finished: ')
        print('Final state: ')
        print(self.env.state)
        print('Final Score: ')
        print(total_reward)
