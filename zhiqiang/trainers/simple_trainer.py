

from . import AbstractTrainer

class SimpleTrainer(AbstractTrainer):
    """
    """
    def __init__(self, settings, agent_class, agent_modules, env_class, buffer_class):
        """
        """
        super(SimpleTrainer, self).__init__()
        self.check_necessary_elements(SimpleTrainer)

        self.settings = settings
        self.agent = agent_class(settings, agent_modules)
        self.agent.env = env_class(settings)
        self.buffer = buffer_class(settings)

    def train(self):
        """
        """
        num_boost = self.settings.trainer_settings["num_boost"]
        num_gen_initial = self.settings.trainer_settings["num_gen_initial"]
        num_gen_increment = self.settings.trainer_settings["num_gen_increment"]
        num_optim = self.settings.trainer_settings["num_optim"]
        #
        batch_size = self.settings.trainer_settings["batch_size"]
        eval_period = self.settings.trainer_settings["eval_period"]
        num_eval_rollout = self.settings.trainer_settings["num_eval_rollout"]
        max_step = self.settings.trainer_settings["max_roll_step"]
        #
        merge_ksi = self.settings.trainer_settings["merge_ksi"]
        max_aver_rewards = self.settings.trainer_settings["base_rewards"]
        list_aver_rewards = []
        #

        # generate experience
        str_info = "initial generating experience ..."
        print(str_info)
        self.settings.logger.info(str_info)
        #
        self.agent.explore_mode()                # explore mode
        count_better = 0
        for idx_gen in range(num_gen_initial):
            experience = self.agent.generate(max_aver_rewards, max_step)
            if len(experience) > 0: count_better += 1
            self.buffer.add(experience)
        #
        str_info = "count_better: %d" % count_better
        print(str_info)
        self.settings.logger.info(str_info)
        #
        # boost      
        for idx_boost in range(num_boost):
            #
            str_info = "-" * 70
            print(str_info)
            self.settings.logger.info(str_info)
            #
            str_info = "curr_boost: %d" % idx_boost
            print(str_info)
            self.settings.logger.info(str_info)
            #
            # eval
            self.agent.eval_mode()                   # eval mode
            if idx_boost % eval_period == 0:
                str_info = "evaluating ..."
                print(str_info)
                self.settings.logger.info(str_info)
                #
                aver_rewards = self.agent.eval(num_eval_rollout, max_step)
                list_aver_rewards.append(aver_rewards)
                #
                str_info = "max_aver_rewards, aver_rewards: %f, %f" % (
                    max_aver_rewards, aver_rewards)
                print(str_info)
                self.settings.logger.info(str_info)
                #
                if aver_rewards >= max_aver_rewards:
                    str_info = "new max_aver_rewards: %f --> %f" % (
                        max_aver_rewards, aver_rewards)
                    print(str_info)
                    self.settings.logger.info(str_info)
                    # update
                    max_aver_rewards = aver_rewards
                    self.agent.update_base_net(merge_ksi)
                    #
            #
            # generate experience
            str_info = "generating experience ..."
            print(str_info)
            self.settings.logger.info(str_info)
            #
            self.agent.explore_mode()                # explore mode
            count_better = 0
            for idx_gen in range(num_gen_increment):
                experience = self.agent.generate(max_aver_rewards, max_step)
                if len(experience) > 0: count_better += 1
                self.buffer.add(experience)
            #
            str_info = "count_better: %d" % count_better
            print(str_info)
            self.settings.logger.info(str_info)
            #
            # optimize
            str_info = "optimizing ..."
            print(str_info)
            self.settings.logger.info(str_info)
            #
            self.agent.train_mode()                  # train mode
            for idx_optim in range(num_optim):
                # sample and standardize
                batch_data = self.buffer.sample(batch_size)
                batch_std = self.agent.standardize_batch(batch_data)
                # optimize
                self.agent.optimize(batch_std, self.buffer)
                #
            #
        #
        # final eval
        #
        str_info = "final evaluating ..."
        print(str_info)
        self.settings.logger.info(str_info)
        #
        self.agent.eval_mode()                       # eval mode
        aver_rewards = self.agent.eval(num_eval_rollout, max_step)
        list_aver_rewards.append(aver_rewards)
        #
        str_info = "aver_rewards: %f" % aver_rewards
        print(str_info)
        self.settings.logger.info(str_info)
        #
        str_info = "finished"
        print(str_info)
        self.settings.logger.info(str_info)
        #
        return list_aver_rewards
        #




    
