

class Settings():
    """
    """
    def __init__(self, file_path):
        """
        """
        self.num_boost = 1000
        self.num_generation = 100
        self.num_train_steps = 100

        #
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

    