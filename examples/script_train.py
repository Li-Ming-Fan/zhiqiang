

import os
from zhiqiang.utils.basic_settings import BasicSettings


#
env_type = "grid_world"
agent_type = "dqn"
settings_filename = "settings_dqn.json"
#
buffer_type = "simple_buffer"
trainer_type = "simple_trainer"
#
##
dir_data_root = "./data_root"
dir_rel_settings = "settings"
#
settings_filepath = os.path.join(dir_data_root, dir_rel_settings, settings_filename)
#
print(os.path.abspath(dir_data_root))
print(settings_filepath)
#

##
if env_type == "grid_world":
    from grid_world import GridWorld as Env
#
if agent_type == "dqn":
    from dqn import DQN as Agent
#

##
if buffer_type == "simple_buffer":
    from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
#
if trainer_type == "simple_trainer":
    from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
#

#
assert False, ""
#

##
settings = BasicSettings(settings_filepath)
settings.dir_base = dir_data_root
settings.dir_rel_settings = dir_rel_settings
#
settings.check_settings()
settings.display()
#
env = Env(settings)
buffer = Buffer(settings)
agent = Agent(settings)
#
trainer = Trainer(settings, agent, env, buffer)
#
trainer.train()
#
