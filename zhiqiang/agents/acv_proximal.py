

from . import AbstractAgent
from ..utils import load_data_from_pkl, save_data_to_pkl

# from zhiqiang.agents import AbstractAgent
# from zhiqiang.utils import load_data_from_pkl, save_data_to_pkl

import numpy as np

import torch
import torch.nn.functional as F


class ProximalACV(torch.nn.Module, AbstractAgent):
    """ Proximal Actor-Critic with V-value Approximation.
    1, proximal optimization for the actor (policy-net),
    2, separate optimization for the actor and the critic.

    """
    def __init__(self, settings, agent_modules, env=None, learning=True):
        """
        """
        super(ProximalACV, self).__init__()
        self.check_necessary_elements(ProximalACV)
        #
        self.settings = settings
        self.pnet_class = agent_modules["pnet"]        
        self.pnet_base = self.pnet_class(self.settings)        
        self.pnet_base.eval_mode()
        self.env = env
        #
        self.policy_greedy = self.settings.agent_settings["policy_greedy"]
        self.policy_epsilon = self.settings.agent_settings["policy_epsilon"]
        self.policy_greedy_bak = self.policy_greedy
        self.policy_epsilon_bak = self.policy_epsilon
        #
        self.num_actions = self.settings.agent_settings["num_actions"]
        self.ppo_epsilon = self.settings.agent_settings["ppo_epsilon"]
        self.gamma = torch.tensor(self.settings.agent_settings["gamma"])
        self.reg_entropy = torch.tensor(self.settings.agent_settings["reg_entropy"])
        #
        self.learning = learning
        if self.learning:
            self.vnet_class = agent_modules["vnet"]
            self.vnet_learner = self.vnet_class(self.settings)
            self.pnet_learner = self.pnet_class(self.settings)
            self.update_base_net(1.0)
            self.merge_ksi = self.settings.agent_settings["merge_ksi"]
        #
    
    #
    def act(self, observation):
        """
        """
        s_std = self.pnet_base.trans_list_observations([observation])
        inference = self.pnet_base.infer(s_std)
        #
        self.action_probs = inference[0]
        if self.policy_greedy:
            return torch.argmax(self.action_probs)
        else:
            return torch.multinomial(self.action_probs, 1)[0]
        #

    def act_with_learner(self, observation):
        """
        """
        s_std = self.pnet_learner.trans_list_observations([observation])
        inference = self.pnet_learner.infer(s_std)
        #
        self.action_probs = inference[0]
        if self.policy_greedy:
            return torch.argmax(self.action_probs)
        else:
            return torch.multinomial(self.action_probs, 1)[0]
        #

    def generate(self, base_rewards, max_step_gen, observation=None):
        """ return: list_experiences, (s, a, r, s', info)
            self.rollout() depends on self.act() and env.
        """
        list_r, list_exp = self.rollout(max_step_gen, observation, "base")
        if sum(list_r) > base_rewards:
            return list_exp
        else:
            return []
        #

    #
    def standardize_batch(self, batch_data):
        """ batch_data["data"]: list of (s, a, r, s', info)
        """
        list_s, list_a, list_r, list_p, list_info = list(zip(*batch_data["data"]))
        #
        s_std = self.pnet_base.trans_list_observations(list_s)
        #
        batch_std = {}
        batch_std["s_std"] = s_std
        batch_std["a"] = torch.tensor(list_a).to(self.settings.device)
        batch_std["r"] = torch.tensor(list_r).to(self.settings.device)
        #
        p_std = self.pnet_base.trans_list_observations(list_p)
        batch_std["p_std"] = p_std
        #
        return batch_std
        #

    def optimize(self, batch_std, buffer):
        """ optimization step
            batch_data["data"]: list of (s, a, r, s', info)
        """
        # s
        s_std = batch_std["s_std"]
        s_v = self.vnet_learner.infer(s_std)            # [B, ]        
        #
        # s'
        p_std = batch_std["p_std"]
        p_v = self.vnet_learner.infer(p_std)            # [B, ]
        #
        ## vnet
        # target
        target = batch_std["r"] + self.gamma * p_v
        target = target.detach()
        #
        # loss
        td_error = target - s_v                         # [B, ]        
        loss = torch.mean(td_error ** 2)
        #
        self.vnet_learner.back_propagate(loss)
        #

        ## pnet
        s_ap_l = self.pnet_learner.infer(s_std)
        indices = batch_std["a"].long().unsqueeze(-1)
        s_exe_ap_l = torch.gather(s_ap_l, 1, indices)        # [B, 1]
        s_exe_ap_l_s = s_exe_ap_l.squeeze(-1)

        s_ap_b = self.pnet_base.infer(s_std)
        s_exe_ap_b = torch.gather(s_ap_b, 1, indices)        # [B, 1]
        s_exe_ap_b_s = s_exe_ap_b.squeeze(-1)
        # s_exe_ap_b_s = s_exe_ap_b_s.detach()

        ## ppo loss
        # td_error = td_error.detach()                          # [B, ]
        log_ap_l = torch.log(s_exe_ap_l_s + 1e-9)             # [B, ]
        #
        ratio = s_exe_ap_l_s / s_exe_ap_b_s                   # [B, ]
        part_1 = ratio * td_error
        part_2 = td_error * torch.clamp(ratio, 1.0-self.ppo_epsilon, 1.0+self.ppo_epsilon)
        target = torch.min(part_1, part_2)         # [B, ]
        target = target.detach()

        # entropy
        log_prob_all = torch.log(s_ap_l + 1e-9)              # [B, NA]
        loss_entropy = torch.sum(s_ap_l * log_prob_all, -1)  # [B, ]
        #
        # loss
        loss = - target * log_ap_l - loss_entropy * self.reg_entropy
        loss = torch.mean(loss)
        #
        self.pnet_learner.back_propagate(loss)
        # self.update_base_net(self.merge_ksi)
        #

    def update_base_net(self, merge_ksi):
        """
        """
        merge_function = self.pnet_base.merge_weights_function()
        merge_function(self.pnet_base, self.pnet_learner, merge_ksi)
        #
    
    def copy_params(self, another):
        """
        """
        merge_function = self.pnet_base.merge_weights_function()
        merge_function(self.pnet_base, another.pnet_base, 1.0)
        if self.learning:
            merge_function(self.pnet_learner, another.pnet_learner, 1.0)
            merge_function(self.vnet_learner, another.vnet_learner, 1.0)
        
    #
    def train_mode(self):
        self.pnet_learner.train_mode()
        self.vnet_learner.train_mode()
        self.policy_greedy = self.policy_greedy_bak
        self.policy_epsilon = self.policy_epsilon_bak

    def eval_mode(self):
        self.pnet_learner.eval_mode()
        # self.vnet_learner.eval_mode()
        self.policy_greedy = 0

    def explore_mode(self):
        # self.pnet_base.eval_mode()
        self.policy_greedy = 0
        self.policy_epsilon = self.policy_epsilon_bak
    #
    def save(self, model_path):
        """
        """
        dict_base = self.pnet_base.state_dict()
        if self.learning:
            dict_pnet_learner = self.pnet_learner.state_dict()
            dict_vnet_learner = self.vnet_learner.state_dict()
        else:
            dict_pnet_learner = {}
            dict_vnet_learner = {}
        #
        dict_all = {"pnet_base": dict_base,
                    "pnet_learner": dict_pnet_learner,
                    "vnet_learner": dict_vnet_learner }
        save_data_to_pkl(dict_all, model_path)
        #

    def load(self, model_path):
        """
        """
        dict_all = load_data_from_pkl(model_path)
        dict_base = dict_all["pnet_base"]
        self.pnet_base.load_state_dict(dict_base)
        if self.learning:
            dict_pnet_learner = dict_all["pnet_learner"]
            dict_vnet_learner = dict_all["vnet_learner"]
            self.pnet_learner.load_state_dict(dict_pnet_learner)
            self.vnet_learner.load_state_dict(dict_vnet_learner)
        #
