

# from abc import ABCMeta, abstractmethod

class AbstractTrainer(object):
    """
    """
    str_sep = "-"*70
    necessary_elements = ["train"]
    necessary_elements_info = """\n%s
    necessary_elements:
    >   function train(self),
        returning nothing.
    \n%s
    """ % (str_sep, str_sep)
    #
    def print_info():
        print(AbstractTrainer.necessary_elements_info)
    #
    def check_necessary_elements(self, subclass_name):
        """
        """
        subclass_elements = subclass_name.__dict__
        for item in AbstractTrainer.necessary_elements:
            if item not in subclass_elements:
                AbstractTrainer.print_info()
                assert False, "function %s NOT implemented. See above for information." % item 
    #
    def __init__(self):
        pass
    #

#
if __name__ == "__main__":

    AbstractTrainer.print_info()
    
