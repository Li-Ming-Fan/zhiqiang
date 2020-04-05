
import numpy as np


from . import AbstractBuffer


class SimpleBuffer(AbstractBuffer):
    """
    """
    def __init__(self, settings):
        """
        """
        self.buffer_size = settings.buffer_settings["buffer_size"]
        self.buffer_list = []
        #
        self.len_buffer = 0
        self.position_list = []
        #

    def _make_ready(self):
        """
        """
        self.len_buffer = len(self.buffer_list)        
        self.position_list = list(range(self.len_buffer))
    
    def add(self, list_experiences):
        """
        """
        len_buffer = len(self.buffer_list)
        len_exp = len(list_experiences)
        #
        if len_buffer + len_exp >= self.buffer_size:
            self.buffer_list[0:len_buffer + len_exp - self.buffer_size] = []
        #
        self.buffer_list.extend(list_experiences)
        #
        self._make_ready()
        #
            
    def sample(self, size, replace=False):
        """ return: dict
        """
        posi = np.random.choice(self.position_list, size, replace=replace)
        #
        data = [self.buffer_list[idx] for idx in posi]
        #
        batch_sample = {"data": data, "position": posi}
        return batch_sample


