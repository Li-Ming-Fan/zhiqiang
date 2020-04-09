

from . import AbstractAgent
from ..utils import load_data_from_pkl, save_data_to_pkl

import numpy as np

import torch
import torch.nn.functional as F


class MStepPolicy(AbstractAgent):
    """
    """
    def __init__(self, settings, agent_modules, env=None, is_learner=True):
        """
        """
        super(MStepPolicy, self).__init__()
        self.check_necessary_elements(MStepPolicy)
        #
        self.settings = settings
        self.pnet_class = agent_modules["pnet"]
        self.pnet_base = self.pnet_class(self.settings.agent_settings)        
        self.pnet_base.eval_mode()
        self.env = env
        #
        self.policy_greedy = self.settings.agent_settings["policy_greedy"]
        self.policy_epsilon = self.settings.agent_settings["policy_epsilon"]
        self.policy_greedy_bak = self.policy_greedy
        self.policy_epsilon_bak = self.policy_epsilon
        #
        self.num_actions = self.settings.agent_settings["num_actions"]
        self.gamma_float = torch.tensor(self.settings.agent_settings["gamma"])
        self.reg_entropy = torch.tensor(self.settings.agent_settings["reg_entropy"])
        self.mstep = self.settings.agent_settings["mstep"]
        #
        self.is_learner = is_learner
        if self.is_learner:
            self.pnet_learner = self.pnet_class(self.settings.agent_settings)
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
        action_probs = inference[0]
        if self.policy_greedy:
            return torch.argmax(action_probs)
        else:
            return torch.multinomial(action_probs, 1)[0]
        #

    def act_with_learner(self, observation):
        """
        """
        s_std = self.pnet_learner.trans_list_observations([observation])
        inference = self.pnet_learner.infer(s_std)
        #
        action_probs = inference[0]
        if self.policy_greedy:
            return torch.argmax(action_probs)
        else:
            return torch.multinomial(action_probs, 1)[0]
        #

    def generate(self, base_rewards, max_step_gen, observation=None):
        """ return: list_experiences, (s, a, r, s', info)
        """
        list_r, list_exp = self.rollout(max_step_gen, observation, "base")
        if sum(list_r) > base_rewards:
            list_mstep_experiences = []
            num_transitions = len(list_exp)
            for idx in range(num_transitions - self.mstep + 1):
                s_start = list_exp[idx][0]
                s_end = list_exp[idx + self.mstep - 1][3]
                s_start_a = list_exp[idx][1]
                #
                r_discounted = 0
                for step in range(self.mstep-1, -1, -1):
                    r_discounted *= self.gamma_float
                    r_discounted += list_exp[idx + step][2]
                #
                mstep_exp = (s_start, s_start_a, r_discounted, s_end, {"mstep": self.mstep})
                list_mstep_experiences.append(mstep_exp)
                #
            #    
            return list_mstep_experiences
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
        batch_std["a"] = torch.tensor(list_a)
        batch_std["r"] = torch.tensor(list_r)
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
        s_ap = self.pnet_learner.infer(s_std)
        indices = batch_std["a"].long().unsqueeze(-1)
        s_exe_ap = torch.gather(s_ap, 1, indices)   # [B, 1]
        #
        # reward
        reward = batch_std["r"]
        reward_with_baseline = reward - torch.mean(reward)    # [B, ]
        #
        
        # loss_pg
        log_ap = torch.log(s_exe_ap.squeeze(-1))       # [B, ]
        # target
        target = F.relu(reward_with_baseline)
        target = target.detach()                        # [B, ]
        #

        # entropy
        log_prob_all = torch.log(s_ap)                     # [B, NA]
        loss_entropy = torch.sum(s_ap * log_prob_all, -1)  # [B, ]
        #
        # loss
        loss = - target * log_ap - loss_entropy * self.reg_entropy
        loss = torch.mean(loss)
        #
        self.pnet_learner.back_propagate(loss)
        self.update_base_net(self.merge_ksi)
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
        if self.is_learner:
            merge_function(self.pnet_learner, another.pnet_learner, 1.0)
        
    #
    def train_mode(self):
        self.pnet_learner.train_mode()
        self.policy_greedy = self.policy_greedy_bak
        self.policy_epsilon = self.policy_epsilon_bak

    def eval_mode(self):
        self.pnet_learner.eval_mode()
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
        if self.is_learner:
            dict_learner = self.pnet_learner.state_dict()
        else:
            dict_learner = {}
        #
        dict_all = {"pnet_base": dict_base, "pnet_learner": dict_learner}
        save_data_to_pkl(dict_all, model_path)
        #

    def load(self, model_path):
        """
        """
        dict_all = load_data_from_pkl(model_path)
        dict_base = dict_all["pnet_base"]
        self.pnet_base.load_state_dict(dict_base)
        if self.is_learner:
            dict_learner = dict_all["pnet_learner"]
            self.pnet_learner.load_state_dict(dict_learner)
        #


