
import numpy as np


from zhiqiang.replay_buffers import AbstractBuffer


class SimpleBuffer(AbstractBuffer):
    """
    """
    def __init__(self, settings):
        """
        """
        self.buffer_size = settings.buffer_settings["buffer_size"]
        self.buffer_list = []
    
    def add(self, list_experience):
        """
        """
        len_buffer = len(self.buffer_list)
        len_exp = len(list_experience)
        #
        if len_buffer + len_exp >= self.buffer_size:
            self.buffer[0:len_buffer + len_exp - self.buffer_size] = []
        #
        self.buffer_list.extend(list_experience)
        #
            
    def sample(self, size):
        """ return: dict
        """
        len_buffer = len(self.buffer_list)
        posi = np.random.sample(list(range(len_buffer)), size)
        #
        data = [self.buffer_list[idx] for idx in posi]
        #
        batch_sample = {"data": data, "position": posi}
        return batch_sample


