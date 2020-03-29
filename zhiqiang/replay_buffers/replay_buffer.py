
import numpy as np


from replay_buffers import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    """
    def __init__(self, settings):
        """
        """
        self.buffer = []
        self.buffer_size = settings.buffer_size
    
    def add(self, experience):
        """
        """
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

