

# from abc import ABCMeta, abstractmethod

class AbstractTrainer(object):
    """
    necessary_elements:
        function do_train(self),
        returning list_aver_rewards.
    """
    necessary_elements = ["do_train"]
    #
    def print_info():
        print("-" * 70)
        print(AbstractTrainer.__doc__)
        print("-" * 70)
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
    
