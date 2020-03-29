

from abc import ABCMeta, abstractmethod

class AbstractBuffer(metaclass=ABCMeta):
    """
    """
    def __init__(self, settings):
        """
        """
        pass

    @abstractmethod    
    def add(self, experience):
        """
        """
        pass

    @abstractmethod             
    def sample(self, num):
        """
        """
        pass

