

import torch
import torch.nn as nn

import numpy as np

import os
import time
from multiprocessing import Process

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
def process_function(model, name):
    print("subprocess id:%d，run：%s" % (os.getpid(), name))
    result = model.infer(model.random_start)
    print(result.size())

#
if __name__ == "__main__":

    # work
    simple_model = SimpleModel()
    result = simple_model.infer(simple_model.random_start)
    print(result.size())  # torch.Size([1, 32, 6, 6]), as expected

    # work
    model_0 = SimpleModel()
    process_function(model_0, "model_0")

    # not work
    model_1 = SimpleModel()
    model_2 = SimpleModel()

    print("main process id：%d" % os.getpid())
    #
    p1 = Process(target=process_function, args=(model_1, "model_1"))
    p2 = Process(target=process_function, args=(model_2, "model_2"))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    #

    # expected result:
    #
    # torch.Size([1, 32, 6, 6])
    # subprocess id:12821，run：model_0
    # torch.Size([1, 32, 6, 6])
    # main process id：12821
    # subprocess id:12834，run：model_1
    # subprocess id:12835，run：model_2
    # torch.Size([1, 32, 6, 6])            # conv1 result
    # torch.Size([1, 32, 6, 6])            # conv1 result

    # actual result:
    #
    # torch.Size([1, 32, 6, 6])
    # subprocess id:12821，run：model_0
    # torch.Size([1, 32, 6, 6])
    # main process id：12821
    # subprocess id:12834，run：model_1
    # subprocess id:12835，run：model_2

    # stuck at conv1 
