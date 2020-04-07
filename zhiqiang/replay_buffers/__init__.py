

class AbstractBuffer(object):
    """
    """
    str_sep = "-"*70
    necessary_elements = ["add", "sample"]
    #
    necessary_elements_info = """\n%s
    necessary_elements:
    >   function add(self, list_experiences),
        returning nothing

        function sample(self, num),
        make a sampling, 
        returning: a dict,
        batch_sample = {"data": [], "position": []}
    \n%s
    """ % (str_sep, str_sep)
    #
    def print_info():
        print(AbstractBuffer.necessary_elements_info)
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
