

from zhiqiang.agents import AbstractAgent

import numpy as np

import torch


class VanilaDQN(AbstractAgent):
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
            return torch.randint(0, self.num_actions, (1,))
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
        s_exe_av = [s_av[idx][a] for idx, a in enumerate(list_a)]
        #
        # s'
        p_std = self.qnet_action.trans_list_observations(list_p)
        p_av = self.qnet_target.infer(p_std)   # [B, A]
        #
        # max value (action) at state s'
        max_p_av = torch.max(p_av, -1)[0]
        #
        # target
        target = torch.tensor(list_r) + self.gamma * max_p_av
        #
        # loss
        loss = target - torch.tensor(s_exe_av)    # [B, ]
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


