

class Settings():
    """
    """
    def __init__(self, file_path=None):
        """
        """
        self.agent_settings = {}
        self.env_settings = {}
        self.buffer_settings = {"buffer_size": 5000}
        self.trainer_settings = {}
        #
        self.trainer_settings["num_boost"] = 1000
        self.trainer_settings["num_gen"] = 100
        self.trainer_settings["num_optim"] = 100
        #
        if file_path is not None:
            self.load_from_json(file_path)
        #

    def save_to_json(self, file_path):
        """
        """
        pass

    def load_from_json(self, file_path):
        """
        """
        pass

    

    