
#
env_type = "grid_world"
agent_type = "dqn"
buffer_type = "replay_buffer"

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


#
from utils import load_settings
#
settings = load_settings(settings_file)
#


#
env = Env(settings)
buffer = Buffer(settings)
#
agent = Agent(settings, env, buffer)
#
agent.train()
#
