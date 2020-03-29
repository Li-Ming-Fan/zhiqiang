

from abc import ABCMeta, abstractmethod

class AbstractAgent(metaclass=ABCMeta):
    """
    """
    def __init__(self, settings):
        """
        """
        pass

    @abstractmethod
    def infer(self, state):
        pass

    @abstractmethod
    def generate(self, env):
        """ generate experience
        """
        pass

    @abstractmethod
    def optimize(self, buffer):
        """ optimization step
        """
        pass

    
