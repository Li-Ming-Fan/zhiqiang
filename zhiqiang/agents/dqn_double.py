

from zhiqiang.agents import AbstractAgent

import numpy as np

import torch


class DoubleDQN(AbstractAgent):
    """
    """
    def __init__(self, settings, qnet_class):
        """
        """
        self.settings = settings
        self.qnet_action = qnet_class(self.settings.agent_settings)
        self.qnet_target = qnet_class(self.settings.agent_settings)
        self.merge_ksi = self.settings.agent_settings["merge_ksi"]
        #
        self.policy_greedy = self.settings.agent_settings["policy_greedy"]
        self.policy_epsilon = self.settings.agent_settings["policy_epsilon"]
        self.num_actions = self.settings.agent_settings["num_actions"]
        self.gamma = torch.tensor(self.settings.agent_settings["gamma"])
        #
        self.max_step_gen = self.settings.agent_settings["max_step_gen"]
        #
    
    #
    def act(self, observation):
        """
        """
        batch_std = self.qnet_action.trans_list_observations([observation])
        inference = self.qnet_action.infer(batch_std)
        #
        action_values = inference[0]
        if self.policy_greedy or np.random.rand() > self.policy_epsilon:
            return torch.argmax(action_values)
        else:
            return torch.randint(0, self.num_actions, (1,))[0]
        #

    #
    def optimize(self, batch_data, buffer):
        """ optimization step
            batch_data["data"]: list of (s, a, r, s', info)
        """
        list_s, list_a, list_r, list_p, list_info = list(zip(*batch_data["data"]))
        #
        # s
        s_std = self.qnet_action.trans_list_observations(list_s)
        s_av = self.qnet_action.infer(s_std)
        indices = torch.LongTensor(list_a).unsqueeze(-1)
        s_exe_av = torch.gather(s_av, 1, indices)
        #
        # s'
        p_std = self.qnet_action.trans_list_observations(list_p)
        #
        # decoupled
        p_av_action = self.qnet_action.infer(p_std)        # [B, A]
        p_action = torch.argmax(p_av_action, -1)           # [B, ]
        #
        p_av_target = self.qnet_target.infer(p_std)        # [B, A]
        indices = torch.LongTensor(p_action).unsqueeze(-1)
        p_exe_av = torch.gather(p_av_target, 1, indices)
        #
        # target
        target = torch.tensor(list_r) + self.gamma * p_exe_av
        target = target.detach()
        #
        # loss
        loss = target - s_exe_av    # [B, ]
        loss = torch.mean(loss ** 2)
        #
        self.qnet_action.back_propagate(loss)
        self.qnet_target.merge_weights(self.qnet_action, self.merge_ksi)
        #

    #
    def prepare_training(self):
        self.qnet_action.prepare_training()

    def prepare_evaluating(self):
        self.qnet_action.prepare_evaluating()
    #


