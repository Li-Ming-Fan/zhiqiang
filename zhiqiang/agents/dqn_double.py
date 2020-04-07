

from . import AbstractAgent

import numpy as np

import torch


class DoubleDQN(AbstractAgent):
    """
    """
    def __init__(self, settings, agent_modules, env=None):
        """
        """
        self.settings = settings
        self.qnet_class = agent_modules["qnet"]
        self.qnet_action = self.qnet_class(self.settings.agent_settings)
        self.qnet_target = self.qnet_class(self.settings.agent_settings)
        self.env = env      
        #
        self.policy_greedy = self.settings.agent_settings["policy_greedy"]
        self.policy_epsilon = self.settings.agent_settings["policy_epsilon"]
        self.policy_greedy_bak = self.policy_greedy
        self.policy_epsilon_bak = self.policy_epsilon
        #
        self.num_actions = self.settings.agent_settings["num_actions"]
        self.gamma = torch.tensor(self.settings.agent_settings["gamma"])
        #
        self.update_base_net(1.0)
        self.qnet_target.eval_mode()
        #
    
    #
    def act(self, observation):
        """
        """
        s_std = self.qnet_action.trans_list_observations([observation])
        inference = self.qnet_action.infer(s_std)
        #
        action_values = inference[0]
        if self.policy_greedy or np.random.rand() > self.policy_epsilon:
            return torch.argmax(action_values)
        else:
            return torch.randint(0, self.num_actions, (1,))[0]
        #

    def generate(self, base_rewards, max_step_gen, observation=None):
        """ return: list_experiences, (s, a, r, s', info)
        """
        sum_rewards, list_experiences = self.rollout(max_step_gen, observation)
        if sum_rewards > base_rewards:
            return list_experiences
        else:
            return []
        #

    #
    def standardize_batch(self, batch_data):
        """ batch_data["data"]: list of (s, a, r, s', info)
        """
        list_s, list_a, list_r, list_p, list_info = list(zip(*batch_data["data"]))
        #
        s_std = self.qnet_action.trans_list_observations(list_s)
        #
        batch_std = {}
        batch_std["s_std"] = s_std
        batch_std["a"] = torch.tensor(list_a)
        batch_std["r"] = torch.tensor(list_r)
        #
        p_std = self.qnet_action.trans_list_observations(list_p)
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
        s_av = self.qnet_action.infer(s_std)
        indices = batch_std["a"].long().unsqueeze(-1)
        s_exe_av = torch.gather(s_av, 1, indices)
        #
        # s'
        p_std = batch_std["p_std"]
        #
        # decoupled
        p_av_action = self.qnet_action.infer(p_std)        # [B, A]
        p_action = torch.argmax(p_av_action, -1)           # [B, ]
        #
        p_av_target = self.qnet_target.infer(p_std)        # [B, A]
        #
        indices = p_action.long().unsqueeze(-1)
        p_exe_av = torch.gather(p_av_target, 1, indices)
        #
        # target
        target = batch_std["r"] + self.gamma * p_exe_av
        target = target.detach()
        #
        # loss
        loss = target - s_exe_av.squeeze(-1)       # [B, ]        
        loss = torch.mean(loss ** 2)
        #
        self.qnet_action.back_propagate(loss)
        #

    def update_base_net(self, merge_ksi):
        """
        """
        merge_function = self.qnet_target.merge_weights_function()
        merge_function(self.qnet_target, self.qnet_action, merge_ksi)
        #
        
    #
    def train_mode(self):
        self.qnet_action.train_mode()
        self.policy_greedy = self.policy_greedy_bak
        self.policy_epsilon = self.policy_epsilon_bak

    def eval_mode(self):
        self.qnet_action.eval_mode()
        self.policy_greedy = 0

    def explore_mode(self):
        self.qnet_action.eval_mode()
        self.policy_greedy = 0
        self.policy_epsilon = self.policy_epsilon_bak
    #


