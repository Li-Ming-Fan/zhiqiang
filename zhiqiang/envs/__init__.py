

class AbstractEnv(object):
    """
    necessary_elements:
        function reset(self),
        returning: observation

        function step(self, action),
        returning: observation, reward, done, info

        function render(self),
        for compatibilty with gym

        function close(self),
        for compatibilty with gym
    
    possible_elements:
        function restore(self, state),
        restore from a given state

        function export(self),
        export the current state
        returning a state
    """
    necessary_elements = ["reset", "step", "render", "close"]
    #
    def print_info():
        print("-" * 70)
        print(AbstractEnv.__doc__)
        print("-" * 70)
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

#
if __name__ == "__main__":

    AbstractEnv.print_info()
    
