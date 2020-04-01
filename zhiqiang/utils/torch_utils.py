
import random
import math
import numpy
import torch

#
def print_params(model, print_name=False):
    """
    """
    n_tr, n_nontr = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape)).item()
        if p.requires_grad:
            n_tr += n_params
        else:
            n_nontr += n_params
    #
    str_info = "n_trainable_params: %d, n_nontrainable_params: %d" % (n_tr, n_nontr)
    print(str_info)
    #
    if print_name:
        for name, value in model.named_parameters():
            if value.requires_grad:
                print("training: %s" % name)
            else:
                print("not training: %s" % name)
    #
    return str_info
    #

def reset_params(model, except_list=[]):
    """
    """
    n_reset, n_unreset = 0, 0
    for name, p in model.named_parameters():
        n_params = torch.prod(torch.tensor(p.shape)).item()
        if name in except_list:        # not reset
            print("not reset: %s" % name)
            n_unreset += n_params
        else:                          # if p.requires_grad:
            if len(p.shape) > 1:
                # torch.nn.init.orthogonal_(p)
                # torch.nn.init.xavier_uniform_(p)
                torch.nn.init.xavier_normal_(p)
                print("reset: %s" % name)
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                print("reset: %s" % name)
            #
            n_reset += n_params
            #
        #
    #
    str_info = "n_reset_params: %d, n_unreset_params: %d" % (n_reset, n_unreset)
    return str_info
    #

def set_random_seed(seed):
    """
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#
def merge_weights(model_a, model_b, merge_ksi):
    """
    """
    pass