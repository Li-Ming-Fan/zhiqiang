

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
        self.agent = agent_class(settings, agent_modules)
        self.agent.env = env_class(settings)
        self.buffer = buffer_class(settings)

        self.num_workers = self.settings.trainer_settings["num_workers"]
        self.list_agents = []
        for idx in range(self.num_workers):
            agent = agent_class(self.settings, agent_modules)
            agent.env = env_class(self.settings)
            self.list_agents.append(agent)

    def train(self):
        """
        """
        num_boost = self.settings.trainer_settings["num_boost"]
        num_gen_initial = self.settings.trainer_settings["num_gen_initial"]
        num_gen_increment = self.settings.trainer_settings["num_gen_increment"]
        num_optim = self.settings.trainer_settings["num_optim"]
        #
        num_workers_initial = self.settings.trainer_settings["num_workers_initial"]
        num_workers_increment = self.settings.trainer_settings["num_workers_increment"]
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
        settings_paral = {"num_workers": num_workers_initial }
        settings_paral["max_aver_rewards"] = max_aver_rewards
        #

        # data_paral_gen
        def process_function_gen(list_data, idx, settings_paral):
            list_result = []
            max_curr = settings_paral["max_aver_rewards"]
            env = self.env_class(self.settings)
            for curr in range(len(list_data)):
                experience = self.agent.generate(max_curr, max_step, env)
                list_result.add(experience)                
            return list_result
        #
        def merge_function_gen(processed_queue_gen, settings_paral):
            list_all = []
            for curr in range(settings_paral["num_workers"]):
                list_all.extend( processed_queue_gen.get() )
            return list_all
        #
        data_paral_gen = DataParallelism(num_workers_initial,
                                         process_function_gen, merge_function_gen)
        #

        # data_paral_eval
        def process_function_eval(list_data, idx, settings_paral):
            list_result = []
            env = self.env_class(self.settings)
            for curr in range(len(list_data)):
                total_reward, list_trans = self.agent.rollout(max_step, env)
                list_result.add(total_reward)
            return list_result
        #
        def merge_function_eval(processed_queue_eval, settings_paral):
            sum_rewards = 0
            for curr in range(settings_paral["num_workers"]):
                sum_curr = sum(processed_queue_eval.get() )
                sum_rewards += sum_curr
            return sum_rewards / num_eval_rollout
        #
        data_paral_eval = DataParallelism(num_workers_initial,
                                          process_function_eval, merge_function_eval)
        #

        # generate experience
        str_info = "initial generating experience ..."
        print(str_info)
        self.settings.logger.info(str_info)
        #
        self.agent.explore_mode()                # explore mode
        #
        data_paral_gen.do_processing(list(range(num_gen_initial)), settings_paral)
        self.buffer.add(data_paral_gen.merged_result)
        count_better = len(data_paral_gen.merged_result)
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
            if idx_boost % eval_period == 0:
                str_info = "evaluating ..."
                print(str_info)
                self.settings.logger.info(str_info)
                #
                self.agent.eval_mode()                # eval mode
                data_paral_eval.do_processing(list(range(num_eval_rollout)), settings_paral)
                aver_rewards = data_paral_eval.merged_result
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
            #
            settings_paral["max_aver_rewards"] = max_aver_rewards
            settings_paral["num_workers"] = num_workers_increment
            data_paral_gen.num_workers = num_workers_increment
            data_paral_gen.reset()
            #
            data_paral_gen.do_processing(list(range(num_gen_increment)), settings_paral)
            self.buffer.add(data_paral_gen.merged_result)
            count_better = len(data_paral_gen.merged_result)
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
        data_paral_eval.do_processing(list(range(num_eval_rollout)), settings_paral)
        aver_rewards = data_paral_eval.merged_result
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




    
