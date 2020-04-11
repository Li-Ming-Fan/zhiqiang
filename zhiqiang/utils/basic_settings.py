
import os
import time
import json
import logging


class BasicSettings():
    """
    """
    def __init__(self, file_path=None):
        """
        """
        self.dir_base = "./data_root"
        self.dir_rel_log = "log"
        self.dir_rel_settings = "settings"
        self.dir_rel_model = "stat_dict"
        self.log_path = None
        self.model_path = None
        self.model_path_timed = None
        #
        self.env = "envname"
        self.agent = "agentname"
        self.buffer = "buffername"
        self.trainer = "trainername"
        #
        self.agent_settings = {}
        self.env_settings = {}
        self.buffer_settings = {"buffer_size": 5000}
        self.trainer_settings = {}
        #
        self.trainer_settings["num_boost"] = 1000
        self.trainer_settings["num_gen"] = 100
        self.trainer_settings["num_optim"] = 100
        #
        self.load_from_json_file(file_path)
        #
        # except from saving
        self.except_list = ["dir_log", "dir_settings", "dir_model", "str_datetime",
               "log_path", "model_path", "model_path_timed", "except_list"]
        #

    #
    def check_settings(self):
        """
        """
        # directories
        self.dir_log = os.path.join(self.dir_base, self.dir_rel_log)
        self.dir_settings = os.path.join(self.dir_base, self.dir_rel_settings)
        self.dir_model = os.path.join(self.dir_base, self.dir_rel_model)
        #        
        if not os.path.exists(self.dir_base): os.mkdir(self.dir_base)
        if not os.path.exists(self.dir_log): os.mkdir(self.dir_log)
        if not os.path.exists(self.dir_settings): os.mkdir(self.dir_settings)
        if not os.path.exists(self.dir_model): os.mkdir(self.dir_model)
        #
        # logger
        self.str_datetime = time.strftime("%Y_%m_%d_%H_%M_%S")
        log_file = "log_%s_%s_%s.txt" % (self.env, self.agent, self.str_datetime)  
        if self.log_path is None:
            self.log_path = os.path.join(self.dir_log, log_file)
        #
        self.logger = self.create_logger(self.log_path)
        print("settings checked")
        #
        self.logger.info(self.trans_info_to_dict())
        #
        # model_path
        model_file = "model_%s_%s_eval.stat_dict" % (self.env, self.agent)
        model_path_eval = os.path.join(self.dir_model, model_file)
        if self.model_path is None:
            self.model_path = model_path_eval
        #
        self.model_path_timed = model_path_eval.replace("eval", self.str_datetime)
        #

    def create_logger(self, log_path):
        """
        """
        logger = logging.getLogger(log_path)  # use log_path as log_name
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding='utf-8') 
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # self.logger.info('test')
        return logger

    def create_or_reset_log_file(self, log_path=None):
        if log_path is None:
            log_path = self.log_path        
        with open(log_path, 'w', encoding='utf-8'):
            print("log file reset")
    
    def close_logger(self, logger=None):
        if logger is None:
            logger = self.logger
        for item in logger.handlers:
            item.close()
            print("logger handler item closed")

    #
    def display(self):
        """
        """        
        print()
        for name, value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            print(str(name) + ': ' + str(value))
        print()
    
    #
    def trans_info_to_dict(self):
        """
        """                
        info_dict = {}
        for name, value in vars(self).items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            if str(name) in self.except_list:
                continue
            info_dict[str(name)] = value        
        return info_dict
    
    def assign_info_from_dict(self, info_dict):
        """
        """
        for key in info_dict:
            value = info_dict[key]
            setattr(self, key, value)
        #

    def assign_info_from_namedspace(self, named_data):
        """
        """
        for key in named_data.__dict__.keys():                 
            self.__dict__[key] = named_data.__dict__[key]
        #
        
    def save_to_json_file(self, file_path):
        """
        """
        info_dict = self.trans_info_to_dict()
        #
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(info_dict, fp, ensure_ascii=False, indent=4)
        #       
    
    def load_from_json_file(self, file_path):
        """
        """
        if file_path is None:
            print("settings file: %s NOT found, using default settings" % file_path)
            return
        #
        with open(file_path, "r", encoding="utf-8") as fp:
            info_dict = json.load(fp)
        #
        self.assign_info_from_dict(info_dict)
        #

   
#       
if __name__ == '__main__':
    
    sett = BasicSettings()
    
    #print(dir(sett))
    #l = [i for i in dir(sett) if inspect.isbuiltin(getattr(sett, i))]
    #l = [i for i in dir(sett) if inspect.isfunction(getattr(sett, i))]
    #l = [i for i in dir(sett) if not callable(getattr(sett, i))]
    
    sett.check_settings()
    sett.display()
    
    print(sett.__dict__.keys())
    print()
    
    info_dict = sett.trans_info_to_dict()
    print(info_dict)
    print()
    
    #
    sett.assign_info_from_dict(info_dict)
    
    #
    info_dict = sett.trans_info_to_dict()
    print(info_dict)
    print()
    #
    
    file_path = os.path.join(sett.dir_settings, "settings_template.json")
    sett.save_to_json_file(file_path)    
    sett.load_from_json_file(file_path)
    
    #
    info_dict = sett.trans_info_to_dict()
    print(info_dict)
    print()
    #

    #    
    sett.close_logger()
    #

    

    

    