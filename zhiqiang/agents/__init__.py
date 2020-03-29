

from abc import ABCMeta, abstractmethod

class AbstractAgent(metaclass=ABCMeta):
    """
    """
    def __init__(self, settings):
        """
        """
        pass

    @abstractmethod
    def generate(self, env):
        """ generate experience
        """
        pass

    @abstractmethod
    def standardize_batch(self, batch_data):
        """ trans batch_data to batch for model
        """
        pass

    @abstractmethod
    def optimize(self, batch):
        """ optimization step
        """
        pass

    
