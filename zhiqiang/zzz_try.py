
from utils import get_package_root_dir

dir_pkg_root = get_package_root_dir()
print(dir_pkg_root)


from utils.basic_settings import BasicSettings
settings = BasicSettings()
settings.env_settings = {}
settings.env_settings["size"] = 5
settings.env_settings["partial"] = False

from envs.grid_world import GridWorld
env = GridWorld(settings)
env.display()


    