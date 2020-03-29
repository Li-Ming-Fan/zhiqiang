
import os

def get_package_root_dir():
    """
    """
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    dir_root = file_dir.rstrip("utils")
    # print(file_path)
    # print(file_dir)
    # print(dir_root)
    return dir_root
    


