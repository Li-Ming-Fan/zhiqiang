# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import os

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
    def __init__(self, num_workers, worker_type = "process"):
        """ worker_type: "thread", or "process"
        """
        self.num_workers = num_workers
        self.worker_type = worker_type  # "process", "thread"

    #
    def do_processing(self, list_data, process_function, merge_function, settings):
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

        # data
        data_split = split_data_list(list_data, self.num_workers)

        # worker
        """
        if self.worker_type == "process":
            worker_class = multiprocessing.Process        # True Process
        else:
            worker_class = multiprocessing.dummy.Process  # A wrapper of Thread
        """
        worker_class = multiprocessing.Process
        print("parent process: %s." % os.getpid())
        #
        self._workers = []
        manager = multiprocessing.Manager()
        with manager:
            self.result_dict = manager.dict()  # 主进程与子进程共享这个字典
            #
            # for idx in range(self.num_workers):
            #     self.result_dict[idx] = []
            #
            for idx in range(self.num_workers):
                #
                p_curr = worker_class(target = process_function,
                                    args = (data_split[idx], idx,
                                            self.result_dict, settings) )
                p_curr.daemon = True
                #
                self._workers.append(p_curr)
                print("worker %d created" % idx)
                #
        
            # process
            for idx in range(self.num_workers):
                self._workers[idx].start()
            #
            for idx in range(self.num_workers):
                self._workers[idx].join()
            #
            print("data processed, begin merging ...")
            
            # merge
            self.merged_result = merge_function(self.result_dict, settings)
            print("result_dict merged.")
            #
        #
        
#
if __name__ == '__main__':

    end_value = 997

    list_data = list(range(end_value))
    settings = {"denominator": 10, "alpha": 0.1, "num_workers": 4}

    direct_result = (0 + end_value-1) *end_value / 2 /10 * 0.1

    ##
    def process_function(list_data, idx, result_dict, settings):
        denom = settings["denominator"]
        result = [item/denom for item in list_data ]
        result_dict[idx] = result

    def merge_function(result_dict, settings):
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
        def __init__(self, settings):
            self.settings = settings

        def main(self, list_data):
            data_paral = DataParallelism(self.settings["num_workers"])
            data_paral.do_processing(list_data, process_function, merge_function,
                                     self.settings)

            print("data_paral result:")
            print(data_paral.merged_result)

            print("direct result:")
            print(direct_result)

    #
    test_p = TestClass(settings)
    test_p.main(list_data)

