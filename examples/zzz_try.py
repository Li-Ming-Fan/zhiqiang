

from zhiqiang.utils.basic_settings import BasicSettings
settings = BasicSettings()
settings.env_settings = {}
settings.env_settings["size"] = 5
settings.env_settings["partial"] = False

from grid_world import GridWorld
env = GridWorld(settings)
env.display()


    