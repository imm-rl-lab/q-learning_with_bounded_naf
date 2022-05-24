import argparse, json, os, random, torch
import numpy as np
from environments.enviroment_generator import generate_env
from models.solver import Solver
from models.agent_generator import AgentGenerator
from models.configure_seed import configure_seed

        
if __name__ == "__main__":
    #get config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as json_config_file:
        config = json.load(json_config_file)

    #set seed
    configure_seed(config.get('random_seed'))

    #get environment
    env = generate_env(config['environment'])

    #get agent
    default_config = {'lr': 1e-3, 'tau': 1e-2, 'gamma': 1, 'epoch_num': 1000, 'batch_size': 128}
    agent_generator = AgentGenerator(env, default_config)
    agent = agent_generator.load(config['model'], default_config)
    agent.noise.threshold = 0

    #get trajectory
    solver = Solver(env)
    solver.evaluate(agent)
