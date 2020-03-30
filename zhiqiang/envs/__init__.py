
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
        """
        """
        pass

    @abstractmethod    
    def step(self, action):
        """
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

