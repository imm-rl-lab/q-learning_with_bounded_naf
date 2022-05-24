import argparse, torch, os, json, random
import matplotlib.pyplot as plt
import numpy as np
from environments.enviroment_generator import generate_env
from models.solver import Solver
from models.agent_generator import AgentGenerator
from models.configure_seed import configure_seed


def plot_reward(epoch_num, rewards, save_plot_path):
    if save_plot_path:
        plt.plot(range(epoch_num), rewards)
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        ax = plt.gca()
        ax.set_facecolor('#eaeaf2')
        plt.grid(color='white')
        plt.savefig(save_plot_path)
        plt.show()
    return None

if __name__ == "__main__":
    #get config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as json_config_file:
        config = json.load(json_config_file)
    
    #get learning config
    learning_config = config['learning']
    
    #set seed
    seed = learning_config.get('random_seed')
    print(f'Start training with random seed: {seed}')
    configure_seed(seed)
    
    #get environment
    env = generate_env(config['environment'])

    #get agent
    agent_generator = AgentGenerator(env, learning_config)
    agent = agent_generator.generate(config['model'])
    
    #train agent
    solver = Solver(env)
    mean_total_rewards = solver.train(agent, learning_config)
    plot_reward(learning_config['epoch_num'], mean_total_rewards, learning_config.get('save_plot_path'))

    #save model
    save_model_path = learning_config.get('save_model_path')
    if save_model_path:
        agent.save(save_model_path)

    #save rewards
    save_rewards_path = learning_config.get('save_rewards_path')
    if save_rewards_path:
        np.save(save_rewards_path, mean_total_rewards)
