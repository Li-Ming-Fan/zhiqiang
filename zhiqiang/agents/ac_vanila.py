

from zhiqiang.agents import AbstractAgent

import numpy as np

import torch


class VanilaAC(AbstractAgent):
    """
    """ 
    def __init__(self, settings, pnet_class, qnet_class):
        """
        """
        self.settings = settings
        self.pnet = pnet_class(self.settings.agent_settings)
        self.qnet = qnet_class(self.settings.agent_settings)        
        #
        self.policy_greedy = self.settings.agent_settings["policy_greedy"]
        self.policy_epsilon = self.settings.agent_settings["policy_epsilon"]
        self.policy_greedy_bak = self.policy_greedy
        self.policy_epsilon_bak = self.policy_epsilon
        #
        self.num_actions = self.settings.agent_settings["num_actions"]
        self.gamma = torch.tensor(self.settings.agent_settings["gamma"])
        #
    
    #
    def act(self, observation):
        """
        """
        s_std = self.pnet.trans_list_observations([observation])
        inference = self.pnet.infer(s_std)
        #
        self.action_probs = inference[0]
        #
        if self.policy_greedy or np.random.rand() > self.policy_epsilon:
            return torch.argmax(self.action_probs)
        else:
            return torch.randint(0, self.num_actions, (1,))[0]
        #

    def rollout(self, max_step, env, observation=None):
        """
        """
        sum_rewards = 0
        list_transitions = []
        #
        if observation is None:
            observation = env.reset()
        #
        for step in range(max_step):
            action = self.act(observation)
            sp, reward, done, info_env = env.step(action)
            #
            info_dict = {"info_env": info_env, "action_probs": self.action_probs}
            exp = (observation, action, reward, sp, info_dict)
            #
            observation = sp
            #
            sum_rewards += reward
            list_transitions.append(exp)
            #
            if done: break
            #
        #
        return sum_rewards, list_transitions
        #

    def generate(self, base_rewards, max_step_gen, env, observation=None):
        """ return: list_experiences, (s, a, r, s', info)
            self.rollout() depends on self.act() and env.
        """
        sum_rewards, list_experiences = self.rollout(max_step_gen, env, observation)
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
        s_std = self.pnet.trans_list_observations(list_s)
        #
        batch_std = {}
        batch_std["s_std"] = s_std
        batch_std["a"] = torch.tensor(list_a)
        batch_std["r"] = torch.tensor(list_r)
        #
        p_std = self.pnet.trans_list_observations(list_p)
        batch_std["p_std"] = p_std
        #

        #
        return batch_std
        #

    def optimize(self, batch_std, buffer):
        """ optimization step
            batch_data["data"]: list of (s, a, r, s', info)
        """
        # s
        s_std = batch_std["s_std"]
        s_v = self.qnet.infer(s_std)            # [B, ]
        #
        # s'
        p_std = batch_std["p_std"]
        p_v = self.qnet.infer(p_std)            # [B, ]
        #
        # target
        target = batch_std["r"] + self.gamma * p_v
        target = target.detach()
        #
        # loss_td
        td_error = target - s_v                 # [B, ]
        loss_td = torch.mean(td_error ** 2)
        #
        # qnet
        self.qnet.back_propagate(loss_td)
        #
        # policy probs
        policy = self.pnet.infer(s_std)                # [B, NA]
        indices = batch_std["a"].long().unsqueeze(-1)
        pp_exe_a = torch.gather(policy, 1, indices)    # [B, 1]
        #
        log_prob = torch.log(pp_exe_a.squeeze(-1))     # [B, ]
        #
        # loss pg
        loss_pg = td_error * log_prob
        #
        self.pnet.back_propagate(loss_pg)
        #

    def update_base_net(self, merge_ksi):
        """
        """
        pass
        # merge_function = self.qnet_target.merge_weights_function()
        # merge_function(self.qnet_target, self.qnet_action, merge_ksi)
        #
        
    #
    def train_mode(self):
        self.pnet.train_mode()
        self.qnet.train_mode()
        self.policy_greedy = self.policy_greedy_bak
        self.policy_epsilon = self.policy_epsilon_bak

    def eval_mode(self):
        self.pnet.eval_mode()
        self.qnet.eval_mode()
        self.policy_greedy = 0

    def explore_mode(self):
        self.pnet.eval_mode()
        self.qnet.eval_mode()
        self.policy_greedy = 0
        self.policy_epsilon = self.policy_epsilon_bak
    #


"""
while True:
      a = actor.act(s)
      sp, r, done, info = env.step(a)
      td_error = critic.learn(s, r, sp)
      actor.learn(s, a, td_error)
      s = sp
"""

