
import numpy as np
import time
import matplotlib.pyplot as plt
import imageio

# option
max_step = 200
time_pause = 0.000001
use_model = False
seed = 10
#

## settings
settings_filepath = "./data_root/settings/settings_atari.json"
#
from zhiqiang.utils.settings_baseboard import SettingsBaseboard
settings = SettingsBaseboard(settings_filepath)
settings.agent_settings["seed"] = seed
#

# env
env_name = "Pong-v0"
#
from atari_env import get_atari_env, display_atari_state, get_num_actions
Env = get_atari_env(env_name, settings)  # num_actions
env = Env(settings)
#
# settings.agent_settings["num_actions"] = get_num_actions(env)
#

#
# actor
#
# random
other_args = get_num_actions(env)
def action_decision(state, other_args):
    a = np.random.randint(0, other_args)
    return a
#
# model
if use_model:
    #
    model_path = "./data_root/stat_dict/model_GridWorld_VanilaDQN_reserved.stat_dict"
    from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
    #
    from gridworld_qnet import GridWorldQNet as QNet
    #
    agent = Agent(settings, {"qnet": QNet}, env=env, learning=False)
    agent.load(model_path)
    #
    # action_decision
    other_args = agent
    def action_decision_model(state, other_args):
        a = agent.act(state)
        a = a.detach().numpy()
        return a
    #
    action_decision = action_decision_model
    #

#
## task-agnostic
#
# env reset
state = env.reset()
#
# rollout
#
list_state = [state ]
list_score = [0 ]
list_action = [-1 ]
#
print("doing rollout ...")
for step in range(max_step):
    #
    # action
    a = action_decision(state, other_args)
    list_action.append(a)
    #
    # step
    state, r, done, info = env.step(a)
    list_state.append(state)
    #
    score = list_score[-1] + r
    list_score.append(score)
    #
    if done: break
    #
#
# display
print("display ...")
gif_images = []
#
plt.ion()
# figure = plt.figure(figsize=(4,3))
for step, item in enumerate(list_state):
    #
    # print(list_action[step])
    #
    pic, picc = display_atari_state(item, show=False, step=step, score=list_score[step])
    gif_images.append(picc)
    #
    plt.pause(time_pause)
#
plt.ioff()
plt.show()
#
imageio.mimsave("./aaa_store/atari_replay_gif.gif", gif_images, fps=10)
#
