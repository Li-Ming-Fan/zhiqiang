
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhiqiang.utils import torch_utils
from zhiqiang.agents import AbstractPQNet

from pong_qnet import ConvModule

#
class PongPNet(torch.nn.Module, AbstractPQNet):
    """ Pong input image: (28, 28, 6)
    """
    def __init__(self, settings):
        """
        """
        super(PongPNet, self).__init__()
        self.check_necessary_elements(PongPNet)
        #
        self.settings = settings
        self.agent_settings = settings.agent_settings
        #
        # conv layers
        self.conv_module = ConvModule(self.settings)
        #
        self.num_actions = self.agent_settings["num_actions"]
        conv_features = self.conv_module.num_conv_features
        num_features = 64
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
        list_s = [np.concatenate(item, -1) for item in list_observation]
        #
        obs_np = np.stack(list_s, axis=0)
        obs_tensor = torch.Tensor(obs_np).to(self.settings.device)
        return obs_tensor

    def infer(self, s_std):
        """ s_std: standardized batch of states
        """
        conv_features = self.conv_module(s_std)
        #
        middle = F.relu(self.linear_0(conv_features))     # [B, M]
        action_values = self.linear_1(middle)             # [B, NA]
        action_probs = torch.softmax(action_values, -1)
        return action_probs

    #
    def merge_weights_function(self):
        return torch_utils.merge_weights
    #


