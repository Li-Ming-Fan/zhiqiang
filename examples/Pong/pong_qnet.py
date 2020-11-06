
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhiqiang.utils import torch_utils
from zhiqiang.agents import AbstractPQNet


class ConvBlockA(torch.nn.Module):
    """
    """
    def __init__(self, input_channel, out_channel):
        """ 3*3, 3*3, 1/2,
        """
        super(ConvBlockA, self).__init__()
        #
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(input_channel, input_channel, 3, padding=1)
        self.convp = nn.Conv2d(input_channel, out_channel, 2, stride=2)
        #

    def forward(self, x):
        """
        """
        c1 = self.conv1(x)
        c1s = c1 + torch.relu(x)
        c2 = self.conv2(c1s)
        c2s = c2 + torch.relu(c1s)
        c3 = self.convp(c2)
        return c3


class ConvModule(torch.nn.Module):
    """
    """
    def __init__(self, settings):
        """
        """
        super(ConvModule, self).__init__()
        self.settings = settings
        self.agent_settings = settings.agent_settings
        #
        fm1 = 64
        #
        # (28, 28, 6)
        self.conv1 = nn.Conv2d(6, 16, (3,3))      # (26, 26, 16)
        self.block1 = ConvBlockA(16, 32)          # (13, 13, 32)
        #
        # self.conv2 = nn.Conv2d(16, 32, (3,3))     # (24, 24, 32)
        # self.block2 = ConvBlockA(32, 64)          # (12, 12, 64)
        #
        self.conv3 = nn.Conv2d(32, 64, (4,4))     # (10, 10, 64)
        self.block3 = ConvBlockA(64, fm1)         # (5, 5, fm1)
        #
        self.list_layers = [self.conv1, self.block1,
                            # self.conv2, self.block2,
                            self.conv3, self.block3 ]
        #
        self.num_conv_features = 5 * 5 * fm1
        #

    def forward(self, s_std):
        """
        """
        batch_permute = s_std.permute(0, 3, 1, 2)    # [B, C, H, W]

        result = batch_permute
        for layer in self.list_layers:
            result = layer(result)
        #
        features = torch.flatten(result, 1)
        #
        return features

#
class PongQNet(torch.nn.Module, AbstractPQNet):
    """ Pong input image: (28, 28, 6)
    """
    def __init__(self, settings):
        """
        """
        super(PongQNet, self).__init__()
        self.check_necessary_elements(PongQNet)
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
        return action_values

    #
    def merge_weights_function(self):
        return torch_utils.merge_weights
    #



