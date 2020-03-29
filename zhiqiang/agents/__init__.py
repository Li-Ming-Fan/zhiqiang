

from abc import ABCMeta, abstractmethod

class AbstractAgent(metaclass=ABCMeta):
    """
    """
    def __init__(self, settings, env, buffer):
        """
        """
        pass

    @abstractmethod
    def infer(self, state):
        pass

    @abstractmethod
    def train(self):
        pass

    
