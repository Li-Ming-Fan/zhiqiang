
import numpy as np


from . import AbstractBuffer


class PriorityBuffer(AbstractBuffer):
    """
    """
    def __init__(self, settings):
        """
        """
        self.buffer_size = settings.buffer_settings["buffer_size"]
        self.buffer_list = []
        self.priority_list = []
        #        
        self.max_priority = 0.0001
        self.len_buffer = 0
        self.position_list = []
        self.probablity_list = []
        #

    def _make_ready(self):
        """
        """
        self.max_priority = max(self.priority_list)
        self.len_buffer = len(self.buffer_list)        
        self.position_list = list(range(self.len_buffer))
        sum_prio = sum(self.priority_list)
        self.probablity_list = [item / sum_prio for item in self.priority_list]
    
    def add(self, list_experiences):
        """ list_experiences: (s, a, r, s', info)
        """
        len_buffer = self.len_buffer
        len_exp = len(list_experiences)
        #
        if len_buffer + len_exp >= self.buffer_size:
            self.buffer_list[0:len_buffer + len_exp - self.buffer_size] = []
            self.priority_list[0:len_buffer + len_exp - self.buffer_size] = []
        #
        self.buffer_list.extend(list_experiences)
        self.priority_list.extend( [ self.max_priority ] * len_exp )
        #
        self._make_ready()
        #
            
    def sample(self, size, replace=False):
        """ return: dict
        """
        posi = np.random.choice(self.position_list, size, replace=replace,
                                p=self.probablity_list)
        #
        dp = [(self.buffer_list[idx], self.priority_list[idx]) for idx in posi]
        data, prio = list(zip(*dp))
        #
        batch_sample = {"data": data, "position": posi, "priority": prio}
        return batch_sample
        #

    def update(self, batch_data):
        """ batch_data: dict
        """
        posi = batch_data["position"]
        prio = batch_data["priority"]
        for idx, posi_c in enumerate(posi):
            self.priority_list[posi_c] = prio[idx]
        #
        self._make_ready()
        #


