

class AbstractEnv(object):
    """
    """
    str_sep = "-"*70
    necessary_elements = ["reset", "step", "render", "close"]
    #
    necessary_elements_info = """\n%s
    necessary_elements:
    >   function reset(self),
        returning: observation

        function step(self, action),
        returning: observation, reward, done, info

        function render(self),
        for compatibilty with gym

        function close(self),
        for compatibilty with gym
    \n
    possible_elements:
    >   function restore(self, state),
        restore from a given state

        function export(self),
        export the current state
        returning a state
    \n%s
    """ % (str_sep, str_sep)
    #
    def print_info():
        print(AbstractEnv.necessary_elements_info)
    #
    def check_necessary_elements(self, subclass_name):
        """
        """
        subclass_elements = subclass_name.__dict__
        for item in AbstractEnv.necessary_elements:
            if item not in subclass_elements:
                AbstractEnv.print_info()
                assert False, "function %s NOT implemented. See above for information." % item 
    #
    def __init__(self):
        pass
    #

