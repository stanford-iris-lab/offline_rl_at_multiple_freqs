from collections import OrderedDict
from copy import deepcopy

from ml_collections import ConfigDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

from .model import Scalar, soft_target_update


class ConservativeDAU(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.use_cql = False
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 5.0
        config.buffer_file = './data0.h5py'
        config.mse_loss = 0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, af, vf, target_af, target_vf):
        self.config = ConservativeDAU.get_default_config(config)
        self.policy = policy
        self.af = af
        self.vf = vf
        self.target_af = target_af
        self.target_vf = target_vf

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.af_optimizer = optimizer_class(
            list(self.af.parameters()), self.config.qf_lr
        )
        self.vf_optimizer = optimizer_class(
            list(self.vf.parameters()), self.config.qf_lr
        )

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.af, self.target_af, soft_target_update_rate)
        soft_target_update(self.vf, self.target_vf, soft_target_update_rate)

    def train(self, batch, discount_arr, n_steps, dt):
        self._total_steps += 1
        batch_size, N, _ = batch['observations'].shape
        n_steps = n_steps.long()-1

        observations = batch['observations'][:,0,:]
        actions = batch['actions'][:,0,:]
        rewards = batch['rewards'].squeeze(-1)
        next_observations = batch['next_observations'].reshape(batch_size*N, -1)
        dones = batch['dones'].squeeze(-1)

        max_actions, log_pi = self.policy(observations)

        """ Policy loss """
        critic_value = self.target_af(observations, max_actions)
        policy_loss = (-critic_value).mean()

        """ Critic loss (updates advantage and value) """
        v = self.vf(observations)
        next_v = (1 - dones).squeeze() * self.target_vf(next_observations)

        pre_adv = self.af(observations, actions)
        pre_max_adv = self.af(observations, max_actions)
        adv = pre_adv - pre_max_adv
        
        # note: reward and discount_arr are already scaled data train loop
        q = v + dt * adv
        expected_q = (rewards.squeeze() * discount_arr * next_v).detach()
        critic_loss = F.mse_loss(q, expected_q)

        ### CQL
        if not self.config.use_cql:
            critic_loss = critic_loss
        else:
            next_observations = batch['next_observations'][np.arange(batch_size), n_steps-1]
            batch_size = actions.shape[0]
            action_dim = actions.shape[-1]
            cql_random_actions = actions.new_empty((batch_size, self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            cql_current_actions, cql_current_log_pis = self.policy(observations, repeat=self.config.cql_n_actions)
            cql_next_actions, cql_next_log_pis = self.policy(next_observations, repeat=self.config.cql_n_actions)
            cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
            cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

            af_pred = self.af(observations, actions)
            cql_af_rand = self.af(observations, cql_random_actions)
            cql_af_current_actions = self.af(observations, cql_current_actions)
            cql_af_next_actions = self.af(observations, cql_next_actions)

            cql_cat_af = torch.cat(
                [cql_af_rand, torch.unsqueeze(af_pred, 1), cql_af_next_actions, cql_af_current_actions], dim=1
            )
            cql_std_af = torch.std(cql_cat_af, dim=1)

            if self.config.cql_importance_sample:
                random_density = np.log(0.5 ** action_dim)
                cql_cat_af = torch.cat(
                    [cql_af_rand - random_density,
                     cql_af_next_actions - cql_next_log_pis.detach(),
                     cql_af_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )
            
            cql_min_af_loss = torch.logsumexp(cql_cat_af / self.config.cql_temp, dim=1).mean() * self.config.cql_min_q_weight * self.config.cql_temp

            """Subtract the log likelihood of data"""
            cql_min_af_loss = cql_min_af_loss - af_pred.mean() * self.config.cql_min_q_weight

            critic_loss = critic_loss + cql_min_af_loss

        self.af_optimizer.zero_grad()
        self.vf_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.af_optimizer.step()
        self.vf_optimizer.step()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )

        if self.config.use_cql:
            cql_metrics = dict(
                cql_std_af=cql_std_af.mean().item(),

                cql_af_rand=cql_af_rand.mean().item(),

                cql_min_af_loss=cql_min_af_loss.mean().item(),

                cql_af_current_actions=cql_af_current_actions.mean().item(),

                cql_af_next_actions=cql_af_current_actions.mean().item(),
            )
        else:
            cql_metrics = {}

        metrics = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            critic_loss=critic_loss.item(),
            q_mean=q.mean().item(),
            average_af=pre_adv.mean().item(),
            total_steps=self.total_steps,
        )
        metrics.update(cql_metrics)
        return metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.af, self.vf, self.target_af, self.target_vf]
        return modules

    @property
    def total_steps(self):
        return self._total_steps
