

from zhiqiang.agents import AbstractAgent


class ModelQNet(object):
    """
    """
    def __init__(self, settings):
        """
        """
        pass



class DQN(AbstractAgent):
    """
    """
    def __init__(self, settings):
        """
        """
        self.settings = settings

    #
    def generate(self, env):
        """ generate experience
        """
        pass

    #
    def standardize_batch(self, batch_data):
        """ trans batch_data to batch for model
        """
        pass

    def optimize(self, batch):
        """ optimization step
        """
        pass



    
