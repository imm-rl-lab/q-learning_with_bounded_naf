import argparse
import json
import os
import random

import numpy as np
import torch

from environments.enviroment_generator import generate_env
from models.agent_evaluation_module import SingleAgentEvaluationModule
from models.agent_generator import AgentGenerator


def configure_random_seed(seed):
    if seed:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=file_path, required=True)
args = parser.parse_args()

with open(args.config) as json_config_file:
    config = json.load(json_config_file)

configure_random_seed(config.get('random_seed'))

env = generate_env(config['environment'])

agent = AgentGenerator(env).load(config['checkpoint'])
env.set_dt(agent.q_model.dt)

evaluation_module = SingleAgentEvaluationModule(env)

evaluation_module.eval_agent(agent)
