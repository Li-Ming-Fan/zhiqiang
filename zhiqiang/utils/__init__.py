
import os
import pickle

##
def save_data_to_pkl(data, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

##
def get_files_with_ext(path, str_ext, flag_walk=False):
    """ get files with filename ending with str_ext, in directory: path
    """
    list_all = []
    if flag_walk:
        # 列出目录下，以及各级子目录下，所有目录和文件
        for (root, dirs, files) in os.walk(path):            
            for filename in files:
                file_path = os.path.join(root, filename) 
                list_all.append(file_path)
    else:
        # 列出当前目录下，所有目录和文件
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            list_all.append(file_path)
    #
    file_list = [item for item in list_all if item.endswith(str_ext)]
    return file_list
    #

##

