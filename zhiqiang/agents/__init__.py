

from abc import ABCMeta, abstractmethod

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
    def generate(self, env):
        """ generate experience
            return: list_experience
        """
        pass

    @abstractmethod
    def standardize_batch(self, batch_data):
        """ trans batch_data to batch_std for model input
            batch_data: dict
            batch_std: dict
        """
        pass

    @abstractmethod
    def optimize(self, batch_std, buffer=None):
        """ optimization step
            batch_std: dict
            buffer: replay_buffer, for possible update
        """
        pass

    
