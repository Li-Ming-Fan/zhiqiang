# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:11:16 2018

@author: limingfan
"""

import os
import multiprocessing

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
    def __init__(self, num_workers, process_function, merge_function,
                 worker_type = "process"):
        """ process_function(list_data, idx, settings),
            returning: processed data, a list
            
            merge_function(processed_queue, settings),
            returning: required form for downstream tasks

            settings: a list, a tuple, a dict or a namespace,
            used in process_function and merge_function

            worker_type: "thread", or "process"
        """
        self.num_workers = num_workers
        self.worker_type = worker_type  # "process", "thread"
        self._process_function_ori = process_function
        self._merge_function_ori = merge_function

        self.reset()

    def reset(self):
        """
        """
        self.processed_queue = multiprocessing.Queue()
        self.merged_result = None

    def _process_function(self, list_data, idx, settings):
        """ wrapper
        """
        list_processed = self._process_function_ori(list_data, idx, settings)
        self.processed_queue.put(list_processed)

    def _merge_function(self, processed_queue, settings):
        """ wrapper
        """
        self.merged_result = self._merge_function_ori(processed_queue, settings)

    #
    def do_processing(self, list_data, settings):
        """ settings: settings used in process_function and merge_function
        """
        # data
        data_split = split_data_list(list_data, self.num_workers)

        # worker
        if self.worker_type == "process":
            from multiprocessing import Process        # True Process
        else:
            from multiprocessing.dummy import Process  # A wrapper of Thread
        self.worker = Process
        #
        print("parent process: %s." % os.getpid())
        #
        lambda_process = lambda data_list, idx, settings: self._process_function(
            data_list, idx, settings)
        #
        self._workers = []
        for idx in range(self.num_workers):
            p_curr = self.worker(target = lambda_process,
                                 args = (data_split[idx], idx, settings) )
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
        self._merge_function(self.processed_queue, settings)
        print("processed_queue merged. all finished.")        

        
    
#
if __name__ == '__main__':

    end_value = 997

    list_data = list(range(end_value))
    settings = {"denominator": 10, "alpha": 0.1, "num_workers": 4}

    direct_result = (0 + end_value-1) *end_value / 2 /10 * 0.1

    ##
    def process_function(list_data, idx, settings):
        denom = settings["denominator"]
        return [item/denom for item in list_data ]

    def merge_function(processed_queue, settings):
        alpha = settings["alpha"]
        sum_all = 0
        for idx in range(settings["num_workers"]):
            result_curr = processed_queue.get()
            sum_curr = sum(result_curr)
            sum_all += sum_curr
        #
        sum_all *= alpha
        return sum_all

    ##
    data_paral = DataParallelism(settings["num_workers"],
                                 process_function, merge_function,
                                 worker_type = "process")
    data_paral.do_processing(list_data, settings)

    print("data_paral result:")
    print(data_paral.merged_result)

    print("direct result:")
    print(direct_result)
