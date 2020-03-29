
from abc import ABCMeta, abstractmethod

class AbstractEnv(metaclass=ABCMeta):
    """
    """
    def __init__(self, settings):
        """
        """
        pass

    @abstractmethod    
    def execute_action(self, action):
        """
        """
        pass

