

from zhiqiang.agents import AbstractAgent

import numpy as np


class DoubleDQN(AbstractAgent):
    """
    """
    def __init__(self, settings, qnet_class):
        """
        """
        self.settings = settings
        self.qnet_action = qnet_class(self.settings.agent_settings)
        self.qnet_target = qnet_class(self.settings.agent_settings)
        self.merge_alpha = self.settings.agent_settings["merge_alpha"]
        #
        self.policy_greedy = self.settings.agent_settings["policy_greedy"]
        self.policy_epsilon = self.settings.agent_settings["policy_epsilon"]
        self.num_actions = self.settings.agent_settings["num_actions"]
        self.gamma = self.settings.agent_settings["gamma"]
        #
        self.max_step_gen = self.settings.agent_settings["max_step_gen"]
        #

    def act(self, observation):
        """
        """
        batch_std = self.qnet_action.trans_list_observation([observation])
        inference = self.qnet_action.infer(batch_std)
        #
        action_values = inference[0]
        if self.policy_greedy or np.random.rand() > self.policy_epsilon:
            return np.argmax(action_values)
        else:
            return np.random.randint(self.num_actions)
        #

    #
    def generate(self, env, observation=None):
        """ generate experience, (s, a, r, s', info)
        """
        list_transition = []
        count = 0
        #
        if observation is None:
            observation = env.reset()
        #
        done = False
        while not done:
            action = self.qnet_action(observation)
            sp, reward, done, info = env.step(action)
            exp = (observation, action, reward, sp, info)
            observation = sp
            #
            list_transition.append(exp)
            count += 1
            #
            if count >= self.max_step_gen: break
            #
        #
        return list_transition

    #
    def optimize(self, batch_data):
        """ optimization step
            batch_data["data"]: list of (s, a, r, s', info)
        """
        list_s, list_a, list_r, list_p, list_info = list(zip(*batch_data["data"]))
        #
        # s
        s_std = self.qnet_action.trans_list_observation(list_s)
        s_av = self.qnet_action.infer(s_std)
        s_exe_av = [s_av[idx][a] for idx, a in enumerate(list_a)]
        #
        # s'
        p_std = self.qnet_action.trans_list_observation(list_p)
        #
        # decoupled
        p_av_action = self.qnet_action.infer(p_std)     # [B, A]
        p_action = np.argmax(p_av_action)               # [B, ]
        #
        p_av_target = self.qnet_target.infer(p_std)     # [B, A]
        p_exe_av = [p_av_target[idx][a] for idx, a in enumerate(p_action)]
        #
        # target
        target = np.array(list_r) + self.gamma * p_exe_av
        #
        # loss
        loss = target - np.array(s_exe_av)    # [B, ]
        loss = np.mean(loss ** 2)
        #
        self.qnet_action.backward(loss)
        self.qnet_target.merge_weights(self.qnet_action, self.merge_alpha)
        #


