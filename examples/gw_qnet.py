

from zhiqiang.agents import AbstractQNet


class GridWorldQNet(AbstractQNet):
    """
    """
    def __init__(self, agent_settings):
        """
        """
        pass

    def trans_list_observation(self, list_observation):
        """ trans list_observation to batch_std for model
            return: batch_std, dict
        """
        pass

    def infer(self, observation):
        """
        """
        pass

    def optimize(self, batch_data):
        """
        """
        pass

    def merge_weights(self, another_qnet, merge_alpha):
        """
        """
        pass


