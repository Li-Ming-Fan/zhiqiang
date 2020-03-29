

import os
from utils import get_package_root_dir
from utils.basic_settings import Settings


#
env_type = "grid_world"
agent_type = "dqn"
buffer_type = "simple_buffer"
trainer_type = "simple_trainer"

dir_data_root = get_package_root_dir()
settings_filename = "file_name"


#
if env_type == "grid_world":
    from envs.grid_world import GridWorld as Env
#
if agent_type == "dqn":
    from agents.dqn import DQN as Agent
#
if buffer_type == "simple_buffer":
    from replay_buffers.simple_buffer import SimpleBuffer as Buffer
#
if trainer_type == "simple_trainer":
    from trainers.simple_trainer import SimpleTrainer as Trainer
#

##
settings_filepath = os.path.join(dir_data_root, "settings", settings_filename)
settings = Settings(settings_filepath)
#
env = Env(settings)
buffer = Buffer(settings)
agent = Agent(settings)
#
trainer = Trainer(settings, agent, env, buffer)
#
trainer.train()
#
