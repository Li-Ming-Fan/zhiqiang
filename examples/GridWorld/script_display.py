
import numpy as np
import time
import matplotlib.pyplot as plt

from zhiqiang.utils.basic_settings import BasicSettings
settings = BasicSettings()
settings.env_settings = {}
settings.env_settings["size"] = 5
settings.env_settings["partial"] = False

from grid_world import GridWorld
env = GridWorld(settings)

#
# plt.ion()
figure_id = env.display()
while True:
    print(figure_id)
    #
    a = np.random.randint(0, 4)
    env.step(a)
    figure_id = env.display(figure_id)
    # time.sleep(1)
    #

