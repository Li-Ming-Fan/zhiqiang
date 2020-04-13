# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import os
import threading
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.multiprocessing as multiprocessing

# 
def split_data_list(list_data, num_split):
    """ list_data: list of data items
        returning: list with num_split elements,
                             each as a list of data items
    """
    num_data_all = len(list_data)
    num_per_worker = num_data_all // num_split
    print("num_data_all: %d" % num_data_all)
    #
    data_split = []
    posi_start = 0
    posi_end = num_per_worker
    for idx in range(num_split):
        list_curr = list_data[posi_start:posi_end]
        data_split.append(list_curr)
        posi_start = posi_end
        posi_end += num_per_worker
    #
    if posi_start < num_data_all:
        data_split[-1].extend(list_data[posi_start:])
    #
    list_num_data = [len(item) for item in data_split]
    print("list_data split: {}".format(list_num_data))
    #
    return data_split
    #

#
class DataParallelism(object):
    """
    """
    def __init__(self, num_workers, process_function, merge_function, settings):
        """ process_function(list_data, idx, result_dict, settings),
            put processed result (a list) into the result_dict,
            result_dict[idx] = list_result 
            returning: nothing
            
            merge_function(result_dict, settings),
            returning: required form for downstream tasks

            settings: a list, a tuple, a dict or a namespace,
            used in process_function and merge_function
        """
        # multiprocessing.set_start_method("spawn", force=True)
        #
        self.num_workers = num_workers
        self.worker_class = multiprocessing.Process
        print("parent process: %s." % os.getpid())
        #
        self.process_function = process_function
        self.merge_function = merge_function
        #
        self.settings = settings
        #
        # input
        self.dict_data_split = {}
        for idx in range(self.num_workers):
            self.dict_data_split[idx] = {"data": []}
        #
        # output
        self.result_dict = multiprocessing.Manager().dict()  # 主进程与子进程共享这个字典 
        #
        # worker
        self._dict_workers = { }
        for idx in range(self.num_workers):
            self._dict_workers[idx] = {}
        #
        for idx in range(self.num_workers):
            p_curr = self.worker_class(target = self.process_function,
                                args = (self.dict_data_split[idx], idx,
                                        self.result_dict, self.settings) )
            p_curr.daemon = True
            #
            self._dict_workers[idx]["worker"] = p_curr
            print("worker %d created" % idx)
            #
        #

    def _rebuild_workers(self):
        """
        """        
        self._rebuild_t = threading.Thread(target=self._rebuild_function)
        self._rebuild_t.daemon = True
        self._rebuild_t.start()
        print("rebuild_thread created and started")
        #
    
    def _rebuild_function(self):
        """
        """
        for idx in range(self.num_workers):
            #
            new_t = self.worker_class(target = self.process_function,
                    args = (self.dict_data_split[idx], idx,
                            self.result_dict, self.settings) )
            new_t.daemon = True
            #
            self._dict_workers[idx]["worker"] = new_t
            print("worker %d rebuilt" % idx)
            #
        #

    #
    def clear_result_dict(self):
        """
        """
        self.result_dict.clear()

    #
    def do_processing(self, list_data, rebuild=False):
        """
        """
        # data
        self.list_data_split = split_data_list(list_data, self.num_workers)

        for idx in range(self.num_workers):
            self.dict_data_split[idx]["data"] = self.list_data_split[idx]     
        
        # process
        for idx in range(self.num_workers):
            self._dict_workers[idx]["worker"].start()

        for idx in range(self.num_workers):
            self._dict_workers[idx]["worker"].join()
        #
        print("data processed, begin merging ...")
        
        # merge
        self.merged_result = self.merge_function(self.result_dict, self.settings)
        print("result_dict merged. all finished.")

        # rebuild
        if rebuild:
            self._rebuild_workers()
        #


##
def process_rt(data_dict, idx, result_dict, settings):
    denom = settings["denominator"]
    result = [item/denom for item in data_dict["data"] ]
    result_dict[idx] = result

def merge_rt(result_dict, settings):
    alpha = settings["alpha"]
    sum_all = 0
    for idx in range(settings["num_workers"]):
        result_curr = result_dict[idx]
        sum_curr = sum(result_curr)
        sum_all += sum_curr
    #
    sum_all *= alpha
    return sum_all

class TestClass(object):
    """
    """
    def __init__(self, settings, process_function, merge_function):
        self.settings = settings
        self.data_paral = DataParallelism(self.settings["num_workers"],
                                          process_function, merge_function,
                                          self.settings)
        
    def main(self, list_data, rebuild=False):

        self.data_paral.do_processing(list_data, rebuild)
        
        print("data_paral result:")
        print(self.data_paral.merged_result)

        


#
if __name__ == '__main__':

    settings = {"denominator": 10, "alpha": 0.1, "num_workers": 4}
    test_p = TestClass(settings, process_rt, merge_rt)

    ##
    end_value = 301

    list_data = list(range(end_value))
    #
    direct_result = (0 + end_value-1) *end_value / 2 /10 * 0.1
    print("direct result:")
    print(direct_result)
    
    test_p.main(list_data, rebuild=True)

    ##
    # import time
    # time.sleep(10)

    end_value = 3013

    list_data = list(range(end_value))
    #
    direct_result = (0 + end_value-1) *end_value / 2 /10 * 0.1
    print("direct result:")
    print(direct_result)
    
    test_p.main(list_data)


    

    

