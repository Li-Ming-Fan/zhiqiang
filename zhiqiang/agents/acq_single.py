

from . import AbstractAgent
from ..utils import load_data_from_pkl, save_data_to_pkl

import numpy as np

import torch
import torch.nn.functional as F


class SingleACQ(AbstractAgent):
    """
    while True:
        a = actor.act(s)
        sp, r, done, info = env.step(a)
        q_sa_target = critic.learn(s, a, r, sp)
        actor.learn(s, a, q_sa_target)
        s = sp
    """
    def __init__(self, settings, agent_modules, env=None, is_learner=True):
        """ A single qnet works as the policy-net and the Q-value net. 
        """
        super(SingleACQ, self).__init__()
        self.check_necessary_elements(SingleACQ)
        #
        self.settings = settings
        self.qnet_class = agent_modules["qnet"]    
        self.qnet_base = self.qnet_class(self.settings.agent_settings)        
        self.qnet_base.eval_mode()
        self.env = env
        #
        self.policy_greedy = self.settings.agent_settings["policy_greedy"]
        self.policy_epsilon = self.settings.agent_settings["policy_epsilon"]
        self.policy_greedy_bak = self.policy_greedy
        self.policy_epsilon_bak = self.policy_epsilon
        #
        self.num_actions = self.settings.agent_settings["num_actions"]
        self.gamma = torch.tensor(self.settings.agent_settings["gamma"])
        self.reg_entropy = torch.tensor(self.settings.agent_settings["reg_entropy"])
        #
        self.is_learner = is_learner
        if self.is_learner:
            self.qnet_learner = self.qnet_class(self.settings.agent_settings)
            self.update_base_net(1.0)
            self.merge_ksi = self.settings.agent_settings["merge_ksi"]
        #
    
    #
    def act(self, observation):
        """
        """
        s_std = self.qnet_base.trans_list_observations([observation])
        inference = self.qnet_base.infer(s_std)
        #
        action_values = inference[0]
        self.action_probs = torch.softmax(action_values, -1)
        if self.policy_greedy:
            return torch.argmax(self.action_probs)
        else:
            return torch.multinomial(self.action_probs, 1)[0]
        #

    def act_with_learner(self, observation):
        """
        """
        s_std = self.qnet_learner.trans_list_observations([observation])
        inference = self.qnet_learner.infer(s_std)
        #
        action_values = inference[0]
        self.action_probs = torch.softmax(action_values, -1)
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
        s_std = self.qnet_base.trans_list_observations(list_s)
        #
        batch_std = {}
        batch_std["s_std"] = s_std
        batch_std["a"] = torch.tensor(list_a)
        batch_std["r"] = torch.tensor(list_r)
        #
        p_std = self.qnet_base.trans_list_observations(list_p)
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
        s_av = self.qnet_learner.infer(s_std)
        indices = batch_std["a"].long().unsqueeze(-1)
        s_exe_av = torch.gather(s_av, 1, indices)        # [B, 1]
        s_exe_av_squeeze = s_exe_av.squeeze(-1)          # [B, ]
        #
        # s'
        p_std = batch_std["p_std"]
        p_av = self.qnet_learner.infer(p_std)            # [B, A]
        #
        # max value (action) at state s'
        max_p_av = torch.max(p_av, -1)[0]
        #
        ## critic
        # target
        target = batch_std["r"] + self.gamma * max_p_av
        target = target.detach()
        #
        # loss
        td_error = target - s_exe_av_squeeze                 # [B, ]        
        loss_critic = torch.mean(td_error ** 2)
        #
        # self.qnet_learner.back_propagate(loss)
        #

        ## actor
        s_ap = torch.softmax(s_av, -1)
        s_exe_ap = torch.gather(s_ap, 1, indices)        # [B, 1]        

        # loss_pg
        log_ap = torch.log(s_exe_ap.squeeze(-1) + 1e-9)     # [B, ]
        # target
        target = F.relu(target)
        # target = target.detach()
        #
        # entropy
        log_prob_all = torch.log(s_ap + 1e-9)               # [B, NA]
        loss_entropy = torch.sum(s_ap * log_prob_all, -1)   # [B, ]
        #
        # reg_pg
        loss_pg = - target * log_ap - loss_entropy * self.reg_entropy
        loss_actor = torch.mean(loss_pg)
        #

        # loss
        loss = loss_critic + loss_actor
        #
        self.qnet_learner.back_propagate(loss)
        # self.update_base_net(self.merge_ksi)
        #

    def update_base_net(self, merge_ksi):
        """
        """
        merge_function = self.qnet_base.merge_weights_function()
        merge_function(self.qnet_base, self.qnet_learner, merge_ksi)
        #
    
    def copy_params(self, another):
        """
        """
        merge_function = self.qnet_base.merge_weights_function()
        merge_function(self.qnet_base, another.qnet_base, 1.0)
        if self.is_learner:
            merge_function(self.qnet_learner, another.qnet_learner, 1.0)
        
    #
    def train_mode(self):
        self.qnet_learner.train_mode()
        self.policy_greedy = self.policy_greedy_bak
        self.policy_epsilon = self.policy_epsilon_bak

    def eval_mode(self):
        self.qnet_learner.eval_mode()
        self.policy_greedy = 0

    def explore_mode(self):
        # self.qnet_base.eval_mode()
        self.policy_greedy = 0
        self.policy_epsilon = self.policy_epsilon_bak
    #
    def save(self, model_path):
        """
        """
        dict_base = self.qnet_base.state_dict()
        if self.is_learner:
            dict_qnet_learner = self.qnet_learner.state_dict()
        else:
            dict_qnet_learner = {}
        #
        dict_all = {"qnet_base": dict_base,
                    "qnet_learner": dict_qnet_learner }
        save_data_to_pkl(dict_all, model_path)
        #

    def load(self, model_path):
        """
        """
        dict_all = load_data_from_pkl(model_path)
        dict_base = dict_all["qnet_base"]
        self.qnet_base.load_state_dict(dict_base)
        if self.is_learner:
            dict_qnet_learner = dict_all["qnet_learner"]
            self.qnet_learner.load_state_dict(dict_qnet_learner)
        #
