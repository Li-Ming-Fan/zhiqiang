
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhiqiang.utils import torch_utils
from zhiqiang.agents import AbstractPQNet


class GridWorldVNet(torch.nn.Module, AbstractPQNet):
    """
    """
    def __init__(self, settings):
        """
        """
        super(GridWorldVNet, self).__init__()
        self.settings = settings
        self.agent_settings = settings.agent_settings
        #
        self.num_actions = self.agent_settings["num_actions"]
        num_features = 64
        # 7*7 --> 5*5 --> 5*5 --> 3*3 --> 3*3
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 1)  # skip
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 1)  # skip, flatten
        self.linear_0 = nn.Linear(64 * 9, num_features)     # 
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
        batch_permute = s_std.permute(0, 3, 1, 2)    # [B, C, H, W]
        c1 = self.conv1(batch_permute)
        c2 = self.conv2(c1)
        c2s = c2 + F.relu(c1)
        c3 = self.conv3(c2s)
        c4 = self.conv4(c3)
        c4s = c4 + F.relu(c3)
        features = torch.flatten(c4s, 1)
        #
        middle = F.relu(self.linear_0(features))       # [B, M]
        action_values = self.linear_1(middle)          # [B, NA]
        state_value = torch.max(action_values, -1)[0]
        return state_value

    #
    def merge_weights_function(self):
        return torch_utils.merge_weights
    #



