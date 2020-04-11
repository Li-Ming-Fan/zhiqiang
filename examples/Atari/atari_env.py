

import gym

#
def get_atari_env(env_name, settings):
    """
    """
    env = gym.make(env_name)
    def env_fn(settings=None):
        return env
    #
    return env_fn
    #

#
atari_env_list = ["CartPole-v1", "SpaceInvaders-v0", "Gopher-v4"]


#
if __name__ == "__main__":
    
    pass

