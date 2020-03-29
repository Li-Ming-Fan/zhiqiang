

from abc import ABCMeta, abstractmethod

class AbstractTrainer(metaclass=ABCMeta):
    """
    """
    def __init__(self, settings, agent, env, buffer):
        """
        """
        pass

    @abstractmethod
    def train(self):
        """
        """
        pass
