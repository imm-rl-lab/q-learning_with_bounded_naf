import torch

from models.ddpg import DDPG
from models.naf import NAF

from models.ou_noise import OUNoise, load_noise
import torch.nn as nn

from models.q_models import QModel, QModel_Bounded, QModel_Bounded_RewardBased, QModel_Bounded_GradientBased
from models.sequential_network import Seq_Network


class AgentGenerator:
    def __init__(self, env, train_settings=None):
        self.dt = env.dt
        self.g = env.g
        if train_settings:
            self.epoch_num = train_settings['epoch_num']
            self.batch_size = train_settings['batch_size']
        else:
            self.epoch_num = 1
            self.batch_size = 64
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_max = env.action_max
        self.action_min = env.action_min
        self.beta = env.beta
        self.r = env.r

        self.noise_min = 1e-2

    def _naf_(self, q_model, model_cfg):
        if model_cfg:
            lr = model_cfg.get('lr', 1e-3)
            gamma = model_cfg.get('gamma', 1)
        else:
            lr = 1e-3
            gamma = 1
        noise = OUNoise(self.action_dim, threshold_min=self.noise_min,
                        threshold_decrease=self.noise_min ** (1 / self.epoch_num))
        return NAF(self.action_min, self.action_max, q_model, noise,
                   batch_size=self.batch_size, gamma=gamma, tau=1e-2, q_model_lr=lr)

    def _generate_naf(self, model_cfg):
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU(), nn.Tanh())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, 128, self.action_dim ** 2], nn.ReLU())
        q_model = QModel(self.action_dim, self.action_min, self.action_max, mu_model, v_model, p_model, self.dt)
        return self._naf_(q_model, model_cfg)

    def _generate_b_naf(self, model_cfg):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, 128, self.action_dim ** 2], nn.ReLU())
        q_model = QModel_Bounded(self.action_dim, self.action_min, self.action_max, nu_model, v_model, p_model, self.dt)
        return self._naf_(q_model, model_cfg)

    def _generate_b_naf_reward_based(self, model_cfg):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_Bounded_RewardBased(self.action_dim, self.action_min, self.action_max, nu_model,
                                             v_model,
                                             self.beta, self.dt)
        return self._naf_(q_model, model_cfg)

    def _generate_b_naf_gradient_based(self, model_cfg):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_Bounded_GradientBased(self.action_dim, self.action_min, self.action_max, v_model, r=self.r,
                                               g=self.g, dt=self.dt)
        return self._naf_(q_model, model_cfg)

    def _generate_ddpg(self, model_cfg):
        if model_cfg:
            lr = model_cfg.get('lr', 1e-3)
            gamma = model_cfg.get('gamma', 1)
        else:
            lr = 1e-3
            gamma = 1
        pi_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        q_model = Seq_Network([self.state_dim + self.action_dim, 256, 128, 1], nn.ReLU())

        noise = OUNoise(self.action_dim, threshold_min=self.noise_min,
                        threshold_decrease=self.noise_min ** (1 / self.epoch_num))
        return DDPG(self.action_min, self.action_max, q_model, pi_model, noise,
                    batch_size=self.batch_size, gamma=gamma, tau=1e-3, q_model_lr=lr)

    def load(self, path, model_cfg=None):
        state_dict = torch.load(path)
        if state_dict['q-model']['model-name'] == 'q-model':
            model = self._generate_naf(model_cfg)
        elif state_dict['q-model']['model-name'] == 'q-model-bounded':
            model = self._generate_b_naf(model_cfg)
        elif state_dict['q-model']['model-name'] == 'q-model-bounded-reward-based':
            model = self._generate_b_naf_reward_based(model_cfg)
        else:
            model = self._generate_b_naf_gradient_based(model_cfg)

        model.q_model.load_state_dict(state_dict['q-model'])
        model.noise = load_noise(state_dict['noise'])
        model.action_min = state_dict['action_min']
        model.action_max = state_dict['action_max']
        model.tau = state_dict['tau']
        model.lr = state_dict['lr']
        model.gamma = state_dict['gamma']
        model.batch_size = state_dict['batch_size']
        return model

    def generate(self, model_cfg):
        model_name = model_cfg['model_name']
        if model_name == 'naf':
            return self._generate_naf(model_cfg)
        elif model_name == 'bnaf':
            return self._generate_b_naf(model_cfg)
        elif model_name == 'rb-bnaf':
            return self._generate_b_naf_reward_based(model_cfg)
        elif model_name == 'gb-bnaf':
            return self._generate_b_naf_gradient_based(model_cfg)
        elif model_name == 'ddpg':
            return self._generate_ddpg(model_cfg)

