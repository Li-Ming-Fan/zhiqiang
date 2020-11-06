
import numpy as np


from . import AbstractBuffer


class FilterBuffer(AbstractBuffer):
    """
    """
    def __init__(self, settings):
        """
        """
        self.buffer_size = settings.buffer_settings["buffer_size"]
        self.important_size = settings.buffer_settings["important_size"]
        self.others_size = self.buffer_size - self.important_size
        #
        self.important_thr = settings.buffer_settings["important_thr"]
        self.important_ratio = settings.buffer_settings["important_ratio"]   
        #
        self.important_buffer = []
        self.others_buffer = []
        self.len_important = 0
        self.len_others = 0
        #
        
    def _make_ready(self):
        """
        """
        self.len_important = len(self.important_buffer)
        self.len_others = len(self.others_buffer)
        self.list_posi_important = list(range(self.len_important))
        self.list_posi_others = list(range(self.len_others))
        print("len_important, len_others: %d, %d" % (
            self.len_important, self.len_others) )

    def filter(self, list_experiences):
        """
        """
        list_important = []
        list_others = []
        for item in list_experiences:
            if item[2] > self.important_thr:
                list_important.append(item)
            else:
                list_others.append(item)
        #
        return list_important, list_others
        #
    
    def add(self, list_experiences):
        """ list_experiences: (s, a, r, s', info)
        """
        list_imp, list_others = self.filter(list_experiences)
        #
        len_imp = len(list_imp)
        len_others = len(list_others)
        #
        # important
        d = len_imp + self.len_important - self.important_size
        if d > 0:
            self.important_buffer[0:d] = []
        #
        self.important_buffer.extend(list_imp)
        #
        # others
        d = len_others + self.len_others - self.others_size
        if d > 0:
            self.others_buffer[0:d] = []
        #
        self.others_buffer.extend(list_others)
        #
        # make_ready
        self._make_ready()
        #
            
    def sample(self, size, replace=False):
        """ return: dict
        """
        imp_size = int(size * self.important_ratio)
        if imp_size > self.len_important:
            imp_size = self.len_important
        #
        others_size = size - imp_size
        #
        posi = np.random.choice(self.list_posi_important, imp_size, replace=replace)
        data_imp = [self.important_buffer[idx] for idx in posi]
        #
        posi_o = np.random.choice(self.list_posi_others, others_size, replace=replace)
        data_others = [self.others_buffer[idx] for idx in posi_o]
        #
        posi_others = [item + self.len_important for item in posi_o]
        #
        data_all = data_imp + data_others
        posi_all = [item for item in posi] + posi_others
        #
        batch_sample = {"data": data_all, "position": posi_all}
        return batch_sample
        #




