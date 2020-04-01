

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
    def generate(self, env, observation=None):
        """ generate experience, (s, a, r, s', info)
            return: list_experience
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
    def rollout(self, env, observation=None):
        """
        """
        rewards = 0
        list_transitions = []
        #
        if observation is None:
            observation = env.reset()
        #
        done = False
        while not done:
            action = self.act(observation)
            sp, reward, done, info = env.step(action)
            exp = (observation, action, reward, sp, info)
            observation = sp
            #
            rewards += reward
            list_transitions.append(exp)
            #
        #
        return rewards, list_transitions
        #

    def eval(self, num_rollout, env, observation=None):
        """
        """
        aver_rewards = 0
        for idx in range(num_rollout):
            rewards, transitions = self.rollout(env)
            aver_rewards += rewards
        #
        return aver_rewards / num_rollout
        #

#
class AbstractQNet(metaclass=ABCMeta):
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
        """ return: action values, numpy.array
        """
        pass

    @abstractmethod
    def back_propagate(self, loss):
        """
        """
        pass

    @abstractmethod
    def merge_weights(self, another_qnet, merge_ksi):
        """
        """
        pass

#
class AbstractPNet(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    

