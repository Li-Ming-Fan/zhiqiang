
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
    params_b = {}
    for name, param in model_b.named_parameters():
        # if param.requires_grad:
        params_b[name] = param.data.clone()

    #
    merge_ksi_comp = 1 - merge_ksi
    #
    param_dict_a = {}
    for name, param in model_a.named_parameters():
        assert name in params_b, "name not exist in model_b"
        p_merged = merge_ksi_comp * param.data + merge_ksi * params_b[name]
        param.data = p_merged
    #

#
class ModelAverage(object):
    """
    """
    def __init__(self, model, decay_ratio):
        """
        """
        self.model = model
        self.decay_ratio = decay_ratio
        self.merge_ratio = 1 - decay_ratio
        self.averaged = {}
        self.backup = {}

    def initialize_averaged(self):
        """
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.averaged[name] = param.data.clone()

    def update_averaged(self):
        """
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.averaged, "not exist in self.averaged"
                p = self.merge_ratio * param.data + self.decay_ratio * self.averaged[name]
                self.averaged[name] = p.clone()
    
    def apply_averaged(self):
        """
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.averaged, "not exist in self.averaged"
                self.backup[name] = param.data
                param.data = self.averaged[name]
    
    def restore_backup(self):
        """
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, "not exist in self.backup"
                param.data = self.backup[name]
        #
        self.backup = {}
        #

#
class GradChecker(object):
    """
    """
    def __init__(self):
        """
        """
        self.grads_dict = {}
    
    def _save_grad(self, name):
        def hook(grad):
            self.grads_dict[name] = grad
        return hook

    def add_to_checker_dict(self, x, name):
        """
        """
        x.register_hook(self._save_grad(name))

    def get_grad(self, name):
        """
        """
        return self.grads_dict[name]
    #
