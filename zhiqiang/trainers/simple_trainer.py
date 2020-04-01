

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
        eval_period = self.settings.trainer_settings["eval_period"]
        num_eval_rollout = self.settings.trainer_settings["num_eval_rollout"]
        list_aver_rewards = []
        #
        for idx_boost in range(num_boost):
            # generate experience
            self.agent.prepare_evaluating()     # eval mode
            for idx_gen in range(num_gen):
                list_experience = self.agent.generate(self.env)
                self.buffer.add(list_experience)
            # eval
            if idx_boost % eval_period == 0:
                aver_rewards = self.agent.eval(num_eval_rollout, self.env)
                list_aver_rewards.append(aver_rewards)
            # optimize
            self.agent.prepare_training()       # train mode
            for idx_optim in range(num_optim):
                batch_data = self.buffer.sample(batch_size)
                self.agent.optimize(batch_data, self.buffer)
        #
        # final eval
        self.agent.prepare_evaluating()         # eval mode
        aver_rewards = self.agent.eval(num_eval_rollout, self.env)
        list_aver_rewards.append(aver_rewards)    
        #
        return list_aver_rewards
        #




    
