

from abc import ABCMeta, abstractmethod

class AbstractBuffer(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    @abstractmethod    
    def add(self, list_experience):
        """
        """
        pass

    @abstractmethod             
    def sample(self, num):
        """ return: dict
            batch_sample = {"data": [], "position": []}
        """ 
        pass

