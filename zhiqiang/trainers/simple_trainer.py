

from zhiqiang.trainers import AbstractTrainer

class SimpleTrainer():
    """
    """
    def __init__(self, settings, agent, env, buffer):
        """
        """
        self.settings = settings
        self.agent = agent
        self.env = env
        self.buffer = buffer

    def train(self):
        """
        """
        num_boost = self.settings.trainer_settings["num_boost"]
        num_gen = self.settings.trainer_settings["num_gen"]
        num_optim = self.settings.trainer_settings["num_optim"]
        #
        batch_size = self.settings.trainer_settings["batch_size"]
        #
        for idx_boost in range(num_boost):
            # generate experience
            for idx_gen in range(num_gen):
                list_experience = self.agent.generate(self.env)
                self.buffer.add(list_experience)
            # optimize
            for idx_optim in range(num_optim):
                batch_data = self.buffer.sample(batch_size)
                self.agent.optimize(batch_data, self.buffer)
            #
        #




    
