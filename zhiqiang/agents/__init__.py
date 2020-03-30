

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
    def eval(self, env, observation=None):
        """
        """
        pass

    #
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
class AbstractQNet(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    @abstractmethod
    def trans_list_observation(self, list_observation):
        """ trans list_observation to batch_std for model
            return: batch_std, dict
        """
        pass

    @abstractmethod
    def infer(self, batch_std):
        """ return: action values, numpy.array
        """
        pass

    @abstractmethod
    def backward(self, loss):
        """
        """
        pass

    @abstractmethod
    def merge_weights(self, another_qnet, merge_alpha):
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

    

