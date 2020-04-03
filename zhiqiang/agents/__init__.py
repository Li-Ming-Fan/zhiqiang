

from abc import ABCMeta, abstractmethod

#
class AbstractAgent(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    @abstractmethod
    def act(self, observation):
        """ choose an action, based on observation
            return: action
        """
        pass

    @abstractmethod
    def optimize(self, batch_data, buffer=None):
        """ optimization step
            batch_data: dict, {"data": data, "position": posi}
            buffer: replay_buffer, for possible update
        """
        pass

    #
    @abstractmethod
    def prepare_training(self):
        """
        """
        pass

    @abstractmethod
    def prepare_evaluating(self):
        """
        """
        pass

    #
    def rollout(self, max_step, env, observation=None):
        """
        """
        rewards = 0
        list_experience = []
        #
        if observation is None:
            observation = env.reset()
        #
        for step in range(max_step):
            action = self.act(observation)
            sp, reward, done, info = env.step(action)
            exp = (observation, action, reward, sp, info)
            observation = sp
            #
            rewards += reward
            list_experience.append(exp)
            #
            if done: break
            #
        #
        return rewards, list_experience
        #

    def eval(self, num_rollout, max_step, env, observation=None):
        """
        """
        aver_rewards = 0
        for idx in range(num_rollout):
            rewards, list_experience = self.rollout(max_step, env)
            aver_rewards += rewards
        #
        return aver_rewards / num_rollout
        #

#
class AbstractPQNet(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    @abstractmethod
    def trans_list_observations(self, list_observations):
        """ trans list_observations to batch_std for model
            return: batch_std, dict
        """
        pass

    @abstractmethod
    def infer(self, batch_std):
        """ return: action_values / policy
        """
        pass

    #
    @abstractmethod
    def prepare_training(self):
        """
        """
        pass

    @abstractmethod
    def prepare_evaluating(self):
        """
        """
        pass

    @abstractmethod
    def back_propagate(self, loss):
        """
        """
        pass

    @abstractmethod
    def merge_weights(self, another_net, merge_ksi):
        """
        """
        pass

