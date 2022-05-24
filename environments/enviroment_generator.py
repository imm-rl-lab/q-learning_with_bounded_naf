from environments.dubinsCar.dubins_car_env import DubinsCar
from environments.pendulum.pendulum_env import Pendulum
from environments.targetProblem.target_problem_env import TargetProblem
from environments.vanDerPol.van_der_pol_env import VanDerPol


def generate_env(config):
    env_name = config['env_name']
    if env_name == 'van-der-pol':
        return VanDerPol(dt=config['dt'])
    elif env_name == 'pendulum':
        return Pendulum(dt=config['dt'])
    elif env_name == 'dubins-car':
        return DubinsCar(dt=config['dt'])
    elif env_name == 'target-problem':
        return TargetProblem(dt=config['dt'])
