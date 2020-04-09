

# define an env
from grid_world import GridWorld as Env

# define a qnet, in PyTorch
from gridworld_qnet import GridWorldQNet as QNet

# pick an agent
from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
from zhiqiang.agents.dqn_double import DoubleDQN as Agent
from zhiqiang.agents.dqn_mstep import MStepDQN as Agent
from zhiqiang.agents.dqn_priority import PriorityDQN as Agent


# pick a buffer
from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
from zhiqiang.replay_buffers.priority_buffer import PriorityBuffer as Buffer

# pick a trainer
from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
# from zhiqiang.trainers.paral_trainer import ParalTrainer as Trainer

# settings file
settings_filepath = "./examples/GridWorld/settings_gridworld.json"

##
#
from zhiqiang.utils.basic_settings import BasicSettings
#
settings = BasicSettings(settings_filepath)
settings.check_settings()
settings.display()
#
trainer = Trainer(settings, Agent, {"qnet": QNet}, Env, Buffer)
#
# train
list_aver_rewards = trainer.do_train()
#
# draw
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 5))
#
eval_period = settings.trainer_settings["eval_period"]
list_x = [idx * eval_period for idx in range(len(list_aver_rewards))]
#
print(list_x)
print(list_aver_rewards)
#
plt.plot(list_x, list_aver_rewards, label="Averaged Rewards", color="r", linewidth=2)
plt.xlabel("Number Boost")
plt.ylabel("Averaged Rewards")  # plt.title("Boost Curriculum")
# plt.xticks(list_x)              # plt.legend()
plt.grid()
plt.show()
