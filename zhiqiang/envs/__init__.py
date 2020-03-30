
from abc import ABCMeta, abstractmethod


class AbstractEnv(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    @abstractmethod    
    def reset(self):
        """ return: observation
        """
        pass

    @abstractmethod    
    def step(self, action):
        """ return: observation, reward, done, info
        """
        pass

    @abstractmethod    
    def render(self):
        """
        """
        pass

    @abstractmethod    
    def close(self):
        """
        """
        pass

