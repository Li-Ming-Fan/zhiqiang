

from . import AbstractTrainer
from ..utils.data_parallelism import DataParallelism

class ParalTrainer(AbstractTrainer):
    """
    """
    def __init__(self, settings, agent_class, agent_modules, env_class, buffer_class):
        """
        """
        super(ParalTrainer, self).__init__()
        self.check_necessary_elements(ParalTrainer)

        self.settings = settings
        self.agent = agent_class(settings, agent_modules, is_learner=True)
        self.agent.env = env_class(settings)
        self.buffer = buffer_class(settings)
        self.agent.train_mode()                  # train mode
        # learner agent (optimization)

        # worker agent (evaluation, exploration)
        self.num_workers = self.settings.trainer_settings["num_workers"]
        self.list_agents = []
        for idx in range(self.num_workers):
            agent = agent_class(self.settings, agent_modules, is_learner=False)
            agent.env = env_class(self.settings)
            self.list_agents.append(agent)

    def set_workers_mode(self, mode):
        """
        """
        for idx in range(self.num_workers):
            if mode == "eval":
                self.list_agents[idx].eval_mode()
            elif mode == "explore":
                self.list_agents[idx].explore_mode()

    def update_workers_params(self):
        """
        """
        for idx in range(self.num_workers):
            self.list_agents[idx].copy_params(self.agent)

    def build_parallelism(self):
        """
        """
        # data_paral_gen
        def process_function_gen(list_data, idx, settings_paral):
            list_result = []
            base_rewards = settings_paral["max_aver_rewards"]
            for curr in range(len(list_data)):
                experience = self.list_agents[idx].generate(base_rewards, self.max_step)
                if len(experience) > 0:
                    list_result.append(experience)  # list of rollouts   
            return list_result
        #
        def merge_function_gen(processed_queue_gen, settings_paral):
            list_all = []
            for curr in range(settings_paral["num_workers"]):
                list_all.extend( processed_queue_gen.get() )
            return list_all
        #
        self.data_paral_gen = DataParallelism(self.settings_paral["num_workers"],
                                        process_function_gen, merge_function_gen)
        #

        # data_paral_eval
        def process_function_eval(list_data, idx, settings_paral):
            list_result = []
            for curr in range(len(list_data)):
                total_reward, list_trans = self.list_agents[idx].rollout(self.max_step)
                list_result.append(total_reward)
            return list_result
        #
        def merge_function_eval(processed_queue_eval, settings_paral):
            sum_rewards = 0
            for curr in range(settings_paral["num_workers"]):
                sum_curr = sum(processed_queue_eval.get() )
                sum_rewards += sum_curr
            return sum_rewards
        #
        self.data_paral_eval = DataParallelism(self.settings_paral["num_workers"],
                                        process_function_eval, merge_function_eval)
        #

    def do_eval(self, num_rollout):
        """
        """
        str_info = "evaluating with %d rollouts ..." % num_rollout
        print(str_info)
        self.settings.logger.info(str_info)
        #
        self.set_workers_mode("eval")                # eval mode
        self.data_paral_eval.reset()
        self.data_paral_eval.do_processing(list(range(num_rollout)), self.settings_paral)
        aver_rewards = self.data_paral_eval.merged_result / num_rollout
        self.list_aver_rewards.append(aver_rewards)
        #
        str_info = "max_aver_rewards, aver_rewards: %f, %f" % (
            self.max_aver_rewards, aver_rewards)
        print(str_info)
        self.settings.logger.info(str_info)
        #
        if aver_rewards >= self.max_aver_rewards:
            str_info = "new max_aver_rewards: %f --> %f" % (
                self.max_aver_rewards, aver_rewards)
            print(str_info)
            self.settings.logger.info(str_info)
            # update
            self.max_aver_rewards = aver_rewards
            self.agent.update_base_net(1.0)            
            #

    def do_exploration(self, num_gen):
        """
        """
        str_info = "generating experience by %d rollouts ..." % num_gen
        print(str_info)
        self.settings.logger.info(str_info)
        #
        self.set_workers_mode("explore")               # explore mode
        self.data_paral_gen.reset()
        self.data_paral_gen.do_processing(list(range(num_gen)), self.settings_paral)
        for item in self.data_paral_gen.merged_result:  # list of rollouts
            self.buffer.add(item)
        #
        count_better = len(self.data_paral_gen.merged_result)
        #
        str_info = "count_better: %d" % count_better
        print(str_info)
        self.settings.logger.info(str_info)

    #
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
        self.max_step = self.settings.trainer_settings["max_roll_step"]
        #
        self.max_aver_rewards = self.settings.trainer_settings["base_rewards"]
        self.list_aver_rewards = []
        #
        self.settings_paral = {"num_workers": self.num_workers }
        self.settings_paral["max_aver_rewards"] = self.max_aver_rewards
        #

        # build paral
        self.build_parallelism()

        # generate experience
        self.do_exploration(num_gen_initial)
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
            if idx_boost % eval_period == 0:
                self.do_eval(num_eval_rollout)
            #
            # generate experience
            self.do_exploration(num_gen_increment)
            #
            # optimize
            str_info = "optimizing ..."
            print(str_info)
            self.settings.logger.info(str_info)
            #
            for idx_optim in range(num_optim):
                # sample and standardize
                batch_data = self.buffer.sample(batch_size)
                batch_std = self.agent.standardize_batch(batch_data)
                # optimize
                self.agent.optimize(batch_std, self.buffer)
                #
            #
            # distribute, update
            self.update_workers_params()
            #
        #
        # final eval
        self.do_eval(num_eval_rollout)
        #
        str_info = "finished"
        print(str_info)
        self.settings.logger.info(str_info)
        #
        return self.list_aver_rewards
        #




    
