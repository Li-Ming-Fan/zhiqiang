

from . import AbstractAgent
from ..utils import load_data_from_pkl, save_data_to_pkl

import numpy as np

import torch


class PriorityDQN(AbstractAgent):
    """
    """
    def __init__(self, settings, agent_modules, env=None, is_learner=True):
        """
        """
        super(PriorityDQN, self).__init__()
        self.check_necessary_elements(PriorityDQN)
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
        #
        self.is_learner = is_learner
        if self.is_learner:
            self.qnet_learner = self.qnet_class(self.settings.agent_settings)
            self.update_base_net(1.0)
            self.merge_ksi = self.settings.agent_settings["merge_ksi"]            
        #
            
    #
    def act(self, observation):
        """ used for play, exploration
        """
        s_std = self.qnet_base.trans_list_observations([observation])
        inference = self.qnet_base.infer(s_std)
        #
        action_values = inference[0]
        if self.policy_greedy or np.random.rand() > self.policy_epsilon:
            return torch.argmax(action_values)
        else:
            return torch.randint(0, self.num_actions, (1,))[0]
        #

    def act_with_learner(self, observation):
        """ used for evaluation of the learner
        """
        s_std = self.qnet_learner.trans_list_observations([observation])
        inference = self.qnet_learner.infer(s_std)
        #
        action_values = inference[0]
        if self.policy_greedy or np.random.rand() > self.policy_epsilon:
            return torch.argmax(action_values)
        else:
            return torch.randint(0, self.num_actions, (1,))[0]
        #

    def generate(self, base_rewards, max_step_gen, observation=None):
        """ generate experiences for training, using qnet_base for rollout
            return: list_experiences, (s, a, r, s', info)
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
        batch_std["position"] = batch_data["position"]
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
        s_exe_av = torch.gather(s_av, 1, indices)
        #
        # s'
        p_std = batch_std["p_std"]
        p_av = self.qnet_base.infer(p_std)   # [B, A]
        #
        # max value (action) at state s'
        max_p_av = torch.max(p_av, -1)[0]
        #
        # target
        target = batch_std["r"] + self.gamma * max_p_av
        target = target.detach()
        #
        # loss
        loss = target - s_exe_av.squeeze(-1)       # [B, ]
        loss_square = loss ** 2      
        loss = torch.mean(loss_square)
        #
        # update priority
        batch_std["priority"] = loss_square.detach().numpy()
        buffer.update(batch_std)
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
            dict_learner = self.qnet_learner.state_dict()
        else:
            dict_learner = {}
        #
        dict_all = {"qnet_base": dict_base, "qnet_learner": dict_learner}
        save_data_to_pkl(dict_all, model_path)
        #

    def load(self, model_path):
        """
        """
        dict_all = load_data_from_pkl(model_path)
        dict_base = dict_all["qnet_base"]
        self.qnet_base.load_state_dict(dict_base)
        if self.is_learner:
            dict_learner = dict_all["qnet_learner"]
            self.qnet_learner.load_state_dict(dict_learner)
        #


