

class AbstractBuffer(object):
    """
    necessary_elements:
        function add(self, list_experiences),
        returning nothing

        function sample(self, num),
        make a sampling, 
        returning: a dict,
        batch_sample = {"data": [], "position": []}
    """
    necessary_elements = ["add", "sample"]
    #
    def print_info():
        print("-" * 70)
        print(AbstractBuffer.__doc__)
        print("-" * 70)
    #
    def check_necessary_elements(self, subclass_name):
        """
        """
        subclass_elements = subclass_name.__dict__
        for item in AbstractBuffer.necessary_elements:
            if item not in subclass_elements:
                AbstractBuffer.print_info()
                assert False, "function %s NOT implemented. See above for information." % item 
    #
    def __init__(self):
        pass
    #

#
if __name__ == "__main__":

    AbstractBuffer.print_info()
    

