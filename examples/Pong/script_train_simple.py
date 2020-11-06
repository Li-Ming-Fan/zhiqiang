

# define an env
from pong import Pong as Env

# define a qnet, in PyTorch
from pong_qnet import PongQNet as QNet
from pong_pnet import PongPNet as PNet
from pong_vnet import PongVNet as VNet
agent_modules = {"pnet": PNet, "qnet": QNet, "vnet": VNet}

# pick an agent
from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
# from zhiqiang.agents.dqn_double import DoubleDQN as Agent
# from zhiqiang.agents.dqn_mstep import MStepDQN as Agent
# from zhiqiang.agents.dqn_priority import PriorityDQN as Agent

# pick a buffer
from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
# from zhiqiang.replay_buffers.priority_buffer import PriorityBuffer as Buffer


# pick a trainer
from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
# from zhiqiang.trainers.paral_trainer import ParalTrainer as Trainer

# settings file, make sure the path is right
settings_filepath = "./data_root/settings/settings_pong.json"
agent_name = "agentname"
env_name = "Pong"

##
#
from zhiqiang.utils.settings_baseboard import SettingsBaseboard
#
settings = SettingsBaseboard(settings_filepath)
settings.env = env_name
settings.agent = agent_name
settings.check_settings()
settings.display()
#
# device
import torch
settings.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
    if settings.device_type is None else torch.device(settings.device_type)
#
print("device: {}".format(settings.device))
#
# trainer
trainer = Trainer(settings, Agent, agent_modules, Env, Buffer)
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
plt.ylabel("Averaged Rewards")    # plt.title("Boost Curriculum")
# plt.xticks(list_x)              # plt.legend()
plt.grid()
plt.show()
