
import numpy as np


from replay_buffers import AbstractBuffer


class SimpleBuffer(AbstractBuffer):
    """
    """
    def __init__(self, settings):
        """
        """        
        self.buffer_size = settings.buffer_settings["buffer_size"]
        self.buffer_list = []
    
    def add(self, experience):
        """
        """
        len_buffer = len(self.buffer_list)
        len_exp = len(experience)
        #
        if len_buffer + len_exp >= self.buffer_size:
            self.buffer[0:len_buffer + len_exp - self.buffer_size] = []
        #
        self.buffer_list.extend(experience)
        #
            
    def sample(self, size):
        """
        """
        return np.random.sample(self.buffer_list, size)


