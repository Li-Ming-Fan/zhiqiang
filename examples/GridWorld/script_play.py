
import numpy as np
import time
import matplotlib.pyplot as plt
import imageio

# option
max_step = 50
use_model = True
seed = 10
#

## settings
settings_filepath = "./data_root/settings/settings_gridworld.json"
#
from zhiqiang.utils.basic_settings import BasicSettings
settings = BasicSettings(settings_filepath)
settings.agent_settings["seed"] = seed
#
# env
from grid_world import GridWorld
env = GridWorld(settings)
#

# actor
#
# random
other_args = None
def action_decision(state, other_args):
    a = np.random.randint(0, 4)
    return a
#
# model
if use_model:
    model_path = "./data_root/zzz_reserved/model_GridWorld_EntropyACV_eval.stat_dict"
    from zhiqiang.agents.acv_entropy import EntropyACV as Agent
    #
    from gridworld_pnet import GridWorldPNet as PNet
    from gridworld_vnet import GridWorldVNet as VNet
    #
    agent = Agent(settings, {"pnet": PNet, "vnet": VNet}, env=env, learning=False)
    agent.load(model_path)
    #
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
#
print("doing rollout ...")
for step in range(max_step):
    #
    # action
    a = action_decision(state, other_args)
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
gif_images = [ ]
#
plt.ion()
figure = plt.figure(figsize=(4,3))
for step, item in enumerate(list_state):
    #
    pic, picc = env.display(item, show=False, step=step, score=list_score[step])
    gif_images.append(picc)
    #
    plt.pause(0.5)
#
plt.ioff()
plt.show()
#
imageio.mimsave("a_replay_gif.gif", gif_images, fps=1)
#
