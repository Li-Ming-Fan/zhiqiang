
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhiqiang.utils import torch_utils
from zhiqiang.agents import AbstractPQNet

from atari_qnet import ConvModule


class AtariVNet(torch.nn.Module, AbstractPQNet):
    """
    """
    def __init__(self, settings):
        """
        """
        super(AtariVNet, self).__init__()
        self.check_necessary_elements(AtariVNet)
        #
        self.settings = settings
        self.agent_settings = settings.agent_settings
        #
        self.num_actions = self.agent_settings["num_actions"]
        conv_features = 64*9
        num_features = 64
        #
        self.conv_module = ConvModule(self.settings)
        #
        self.linear_0 = nn.Linear(conv_features, num_features)     # 
        self.linear_1 = nn.Linear(num_features, self.num_actions)
        #
        # optimizer
        params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.agent_settings["lr"],
                                          weight_decay=self.agent_settings["l2reg"])
        #
        self.reset(seed=self.agent_settings["seed"])
        #

    def reset(self, seed=100):
        """
        """
        torch_utils.set_random_seed(seed)
        torch_utils.reset_params(self)
        torch_utils.print_params(self)

    #
    def trans_list_observations(self, list_observation):
        """ trans list_observation to batch_std for model
            return: s_std, standardized batch of states
        """
        obs_np = np.stack(list_observation, axis=0)
        obs_tensor = torch.Tensor(obs_np).to(self.settings.device)
        return obs_tensor

    def infer(self, s_std):
        """ s_std: standardized batch of states
        """
        conv_features = self.conv_module(s_std)
        #
        middle = F.relu(self.linear_0(conv_features))     # [B, M]
        action_values = self.linear_1(middle)             # [B, NA]
        state_value = torch.max(action_values, -1)[0]
        return state_value

    #
    def merge_weights_function(self):
        return torch_utils.merge_weights
    #



