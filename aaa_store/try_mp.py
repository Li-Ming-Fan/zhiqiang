

import torch
import torch.nn as nn

import numpy as np

import os
import time

from zhiqiang.utils.data_parallelism import DataParallelism

# define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.random_start = torch.tensor(np.ones([1, 3, 7, 7]), dtype=torch.float32)
        self.conv1 = nn.Conv2d(3, 32, 2)

    def infer(self, input_batch):
        conved = self.conv1(input_batch)
        return conved

# define subprocess run function
def process_function(list_data, idx, queue, settings):
    """
    """
    model = settings["model"][idx]
    name = settings["name"][idx]
    print("subprocess id:%d，run：%s" % (os.getpid(), name))
    result = model.infer(model.random_start)
    print(result.size())

def merge_function(queue, settings):
    pass

#
if __name__ == "__main__":

    # work
    simple_model = SimpleModel()
    result = simple_model.infer(simple_model.random_start)
    print(result.size())  # torch.Size([1, 32, 6, 6]), as expected

    # work
    model_0 = SimpleModel()

    settings = {"model": [model_0], "name": ["model_0"]}
    process_function([], 0, None, settings)

    #
    model_1 = SimpleModel()
    model_2 = SimpleModel()

    import multiprocessing as mp
    mp.set_start_method("spawn")

    print("main process id：%d" % os.getpid())
    #
    settings = {"model": [model_1, model_2],
                "name": ["model_1", "model_2"]}

    data_paral = DataParallelism(2)
    data_paral.do_processing([0, 1], process_function, merge_function, settings)

    