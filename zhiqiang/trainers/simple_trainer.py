

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

        self.num_boost = self.settings.trainer_settings["num_boost"]
        self.num_gen_initial = self.settings.trainer_settings["num_gen_initial"]
        self.num_gen_increment = self.settings.trainer_settings["num_gen_increment"]
        self.num_optim = self.settings.trainer_settings["num_optim"]
        #
        self.batch_size = self.settings.trainer_settings["batch_size"]
        self.eval_period = self.settings.trainer_settings["eval_period"]
        self.num_eval_rollout = self.settings.trainer_settings["num_eval_rollout"]
        self.max_step = self.settings.trainer_settings["max_roll_step"]
        #
        self.max_aver_rewards = self.settings.trainer_settings["base_rewards"]
        self.list_aver_rewards = []
        #

    def log_info(self, str_info):
        """
        """
        print(str_info)
        self.settings.logger.info(str_info)

    #
    def do_eval(self, num_rollout):
        """
        """
        self.log_info("evaluating with %d rollouts ..." % num_rollout)
        #
        self.agent.load(self.settings.model_path)
        self.agent.eval_mode()
        aver_rewards = self.agent.eval(num_rollout, self.max_step)
        #
        self.log_info("eval aver_rewards: %f" % aver_rewards)
        #
        return aver_rewards

    def do_play(self, num_times):
        """
        """
        self.agent.load(self.settings.model_path)
        self.agent.eval_mode()
        pass

    #
    def _eval_for_train(self, num_rollout):
        """
        """
        self.log_info("evaluating with %d rollouts ..." % num_rollout)
        #
        self.agent.eval_mode()                   # eval mode
        aver_rewards = self.agent.eval(num_rollout, self.max_step)        
        #
        self.log_info("max_aver_rewards, aver_rewards: %f, %f" % (
            self.max_aver_rewards, aver_rewards) )
        #
        # check
        if aver_rewards >= self.max_aver_rewards:
            self.log_info("new max_aver_rewards: %f --> %f" % (
                self.max_aver_rewards, aver_rewards) )
            # update
            self.max_aver_rewards = aver_rewards
            self.agent.update_base_net(1.0)
            # save
            self.agent.save(self.settings.model_path)
            self.agent.save(self.settings.model_path_timed)
        #
        return aver_rewards
        #

    def _explore_for_train(self, num_gen):
        """
        """
        self.log_info("generating experience by %d rollouts ..." % num_gen)
        #
        self.agent.explore_mode()                # explore mode
        count_better = 0
        for idx_gen in range(num_gen):
            experience = self.agent.generate(self.max_aver_rewards, self.max_step)
            if len(experience) > 0:
                count_better += 1
                self.buffer.add(experience)
        #
        self.log_info("count_better: %d" % count_better)

    #
    def do_train(self, model_path=None):
        """
        """
        # load
        if model_path is not None:
            self.agent.load(self.settings.model_path)

        # generate experience
        self._explore_for_train(self.num_gen_initial)
        #
        # boost      
        for idx_boost in range(self.num_boost):
            #
            self.log_info("-" * 70)
            self.log_info("curr_boost: %d" % idx_boost)
            #
            # eval            
            if idx_boost % self.eval_period == 0:
                aver_rewards = self._eval_for_train(self.num_eval_rollout)
                self.list_aver_rewards.append(aver_rewards)
            #
            # generate experience
            self._explore_for_train(self.num_gen_increment)
            #
            # optimize
            self.log_info("optimizing ...")
            #
            self.agent.train_mode()                  # train mode
            for idx_optim in range(self.num_optim):
                # sample and standardize
                batch_data = self.buffer.sample(self.batch_size)
                batch_std = self.agent.standardize_batch(batch_data)
                # optimize
                self.agent.optimize(batch_std, self.buffer)
                #
            #
        #
        # final eval
        self.log_info("-" * 70)
        aver_rewards = self._eval_for_train(self.num_eval_rollout)
        self.list_aver_rewards.append(aver_rewards)
        #
        self.log_info("finished")
        #
        return self.list_aver_rewards
        #




    
