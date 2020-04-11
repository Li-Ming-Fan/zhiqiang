

import gym
import matplotlib.pyplot as plt
import imageio
import os



#
atari_env_list = ["AirRaid-v0", "Alien-v0", "Amidar-v0",
                  "Assault-v0", "Asterix-v0", "Asteroids-v0",
                  "Atlantis-v0", "BankHeist-v0", "BattleZone-v0",
                  "BeamRider-v0", "Berzerk-v0", "Bowling-v0",
                  "Boxing-v0", "Breakout-v0", "Carnival-v0",
                  "Centipede-v0", "ChopperCommand-v0", "CrazyClimber-v0",
                  "DemonAttack-v0", "DoubleDunk-v0", "ElevatorAction-v0",
                  "Enduro-v0", "FishingDerby-v0", "Freeway-v0",
                  "Frostbite-v0", "Gopher-v0", "Gravitar-v0",
                  "IceHockey-v0", "Jamesbond-v0", "JourneyEscape-v0",
                  "Kangaroo-v0", "Krull-v0", "KungFuMaster-v0",
                  "MontezumaRevenge-v0", "MsPacman-v0", "NameThisGame-v0",
                  "Phoenix-v0", "Pitfall-v0", "Pong-v0",
                  "Pooyan-v0", "PrivateEye-v0", "Qbert-v0",
                  "Riverraid-v0", "RoadRunner-v0", "Robotank-v0",
                  "Seaquest-v0", "Skiing-v0", "Solaris-v0",
                  "SpaceInvaders-v0", "StarGunner-v0", "Tennis-v0",
                  "TimePilot-v0", "Tutankham-v0", "UpNDown-v0",
                  "Venture-v0", "VideoPinball-v0", "WizardOfWor-v0",
                  "YarsRevenge-v0", "Zaxxon-v0" ]

#
def get_atari_env(env_name, settings):
    """
    """
    env = gym.make(env_name)
    settings.agent_settings["num_actions"] = env.action_space.n
    #
    def env_fn(settings=None):
        return env
    #
    return env_fn

def get_num_actions(env):
    return env.action_space.n

#
def display_atari_state(state, show=True, step=None, score=None):
    """
    """
    plt.imshow(state, interpolation="nearest")
    if step is not None:
        if score is not None:
            plt.title("step: %d, score: %f" % (step, score))
        else:
            plt.title("step: %d" % (step, ))
    else:
        if score is not None:
            plt.title("score: %f" % (score, ))
        #
    #
    if show:
        plt.show()
    #
    filename = "atari_temp.eps"  # eps < png < jpg
    plt.savefig(filename)
    pic_with_score = imageio.imread(filename) 
    os.remove(filename)
    #
    return state, pic_with_score
    #


#
if __name__ == "__main__":

    import gym
    env = gym.make("Pong-v0")
    print(env.action_space)
    #> Discrete(6)
    print(env.observation_space)
    #> Box(210, 160, 3)

    print(env.action_space.n)
    print(get_num_actions(env))


