

# define an env
from grid_world import GridWorld as Env
#
# define a qnet, in PyTorch or Tensorflow or others
from gw_qnet import GridWorldQNet as QNet
#
# pick an agent
from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
#
# pick a buffer
from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
#
# pick a trainer
from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
#
# settings file
settings_filepath = "./data_root/settings/settings_dqn_vanila.json"
#

# The following is a common routine.
from zhiqiang.utils.basic_settings import BasicSettings
#
settings = BasicSettings(settings_filepath)
settings.check_settings()
settings.display()
#
env = Env(settings)
buffer = Buffer(settings)
agent = Agent(settings, QNet)
trainer = Trainer(settings, agent, env, buffer)
#
# train
list_aver_rewards = trainer.train()
#
# draw
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 5))
#
eval_period = settings.trainer_settings["eval_period"]
list_x = [idx * eval_period for idx in range(len(list_aver_rewards))]
#
plt.plot(num_hop, list_aver_rewards, label="Averaged Rewards", color="k", linewidth=2)
plt.xlabel("Number Boost")
plt.ylabel("Averaged Rewards")  # plt.title("Boost curriculum")
plt.xticks(list_x)              # plt.legend()
plt.grid()
plt.show()
