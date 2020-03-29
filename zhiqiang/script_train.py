
#
env_type = "grid_world"
agent_type = "dqn"
buffer_type = "replay_buffer"
trainer_type = "simple_trainer"


settings_file = "file_path"


#
if env_type == "grid_world":
    from envs.grid_world import GridWorld as Env
#
if agent_type == "dqn":
    from agents.dqn import DQN as Agent
#
if buffer_type == "replay_buffer":
    from replay_buffers.replay_buffer import ReplayBuffer as Buffer
#
if trainer_type == "simple_trainer":
    from trainers.simple_trainer import SimpleTrainer as Trainer
#

#
from utils.basic_settings import Settings
#
settings = Settings(settings_file)
#


#
env = Env(settings)
buffer = Buffer(settings)
agent = Agent(settings)
#
trainer = Trainer(settings, agent, env, buffer)
#
trainer.train()
#
