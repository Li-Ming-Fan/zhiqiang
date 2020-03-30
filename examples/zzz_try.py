

from zhiqiang.utils.basic_settings import BasicSettings
settings = BasicSettings()
settings.env_settings = {}
settings.env_settings["size"] = 5
settings.env_settings["partial"] = False

from grid_world import GridWorld
env = GridWorld(settings)
# env.display()


import numpy as np

print(np.random.rand())
print(np.random.random())


batch_data = {"data": 0}
s_std = {"s": 1}
p_std = {"p": 2}

batch_std = dict(batch_data, **s_std, **p_std)

print(batch_std)

