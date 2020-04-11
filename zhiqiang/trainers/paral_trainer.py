

from . import AbstractTrainer
from ..utils.data_parallelism import split_data_list

import os
import multiprocessing as mp

#
class WorkerProcess(mp.Process):
    """
    """
    def __init__(self, settings, agent_class, agent_modules, env_class):
        """
        """
        super(WorkerProcess, self).__init__()
        #
        self.settings = settings
        self.agent = agent_class(settings, agent_modules, is_learner=True)
        self.agent.env = env_class(settings)
        #
        self.mode = "explore"  # "eval"
        self.list_data = []
        self.settings_paral = {}
        #
        self.result = None
        #

    def run(self):
        """
        """
        if self.mode == "explore":
            self.process_gen(self.list_data, self.settings_paral)
        elif self.mode == "eval":
            self.process_eval(self.list_data, self.settings_paral)
        #

    #
    def process_eval(self, list_data, settings_paral):
        max_step = settings_paral["max_step"]
        #
        list_result = []
        for curr in range(len(list_data)):
            total_reward, list_trans = self.agent.rollout(max_step)
            list_result.append(total_reward)
        #
        self.result = list_result
        #

    #
    def process_gen(self, list_data, settings_paral):
        max_step = settings_paral["max_step"]
        base_rewards = settings_paral["max_aver_rewards"]
        #
        list_result = []
        for curr in range(len(list_data)):
            expr = self.agent.generate(base_rewards, max_step)
            if len(expr) > 0:
                list_result.append(expr)  # list of experiences, training sample   
        #
        self.result = list_result
        #
    #

#
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
        #
        self.list_workers = []
        for idx in range(self.num_workers):
            worker = WorkerProcess(settings, agent_class, agent_modules, env_class)
            self.list_workers.append(worker)
        #

        #
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
        # import multiprocessing as mp
        mp.set_start_method("fork")
        #

    def log_info(self, str_info):
        """
        """
        print(str_info)
        self.settings.logger.info(str_info)

    def set_workers_mode(self, mode):
        """
        """
        self.settings_paral = {"num_workers": self.num_workers }
        self.settings_paral["max_aver_rewards"] = self.max_aver_rewards
        self.settings_paral["max_step"] = self.max_step
        #
        for idx in range(self.num_workers):
            self.list_workers[idx].settings_paral = self.settings_paral
            #
            if mode == "eval":
                self.list_workers[idx].agent.eval_mode()
                self.list_workers[idx].mode = "eval"
            elif mode == "explore":
                self.list_workers[idx].agent.explore_mode()
                self.list_workers[idx].mode = "explore"
            #

    def update_workers_params(self):
        """
        """
        for idx in range(self.num_workers):
            self.list_workers[idx].agent.copy_params(self.agent)

    #
    def do_eval(self, num_rollout):
        """
        """
        self.log_info("evaluating with %d rollouts ..." % num_rollout)
        aver_rewards = self._eval_procedure(num_rollout)
        self.log_info("eval aver_rewards: %f" % aver_rewards)
        #
        return aver_rewards

    #
    def _eval_procedure(self, num_rollout):
        """
        """
        list_split = split_data_list(list(range(num_rollout)), self.num_workers)
        #
        self.set_workers_mode("eval")                # eval mode
        for idx in range(self.num_workers):
            self.list_workers[idx].list_data = list_split[idx]
            self.list_workers[idx].start()
        for idx in range(self.num_workers):
            self.list_workers[idx].join()
        #
        merged_result = self._merge_eval_result()
        aver_rewards = merged_result / num_rollout
        #
        return aver_rewards
        #

    #
    def _eval_for_train(self, num_rollout):
        """
        """
        self.log_info("evaluating with %d rollouts ..." % num_rollout)
        aver_rewards = self._eval_procedure(num_rollout)
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

    def _explore_for_train(self, num_rollout):
        """
        """
        self.log_info("generating experience by %d rollouts ..." % num_rollout)
        #
        list_split = split_data_list(list(range(num_rollout)), self.num_workers)
        #
        self.set_workers_mode("explore")               # explore mode
        for idx in range(self.num_workers):
            self.list_workers[idx].list_data = list_split[idx]
            self.list_workers[idx].start()
        for idx in range(self.num_workers):
            self.list_workers[idx].join()
        #
        merged_result = self._merge_gen_result()
        #
        for item in merged_result:     # list of experiences
            self.buffer.add(item)
        #
        count_better = len(merged_result)
        #
        self.log_info("count_better: %d" % count_better)
        #

    #
    def _merge_eval_result(self):
        """
        """
        sum_rewards = 0
        for curr in range(self.num_workers):
            sum_curr = sum(self.list_workers[curr].result)
            sum_rewards += sum_curr
        #
        return sum_rewards
    #
    def _merge_gen_result(self):
        """
        """
        list_all = []
        for curr in range(self.num_workers):
            list_all.extend(self.list_workers[curr].result)
        #
        return list_all         # list of experiences
    #

    #
    def do_train(self, model_path=None):
        """
        """
        # load
        if model_path is not None:
            self.agent.load(self.settings.model_path)
            self.update_workers_params()

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
            for idx_optim in range(self.num_optim):
                # sample and standardize
                batch_data = self.buffer.sample(self.batch_size)
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
        self.log_info("-" * 70)
        aver_rewards = self._eval_for_train(self.num_eval_rollout)
        self.list_aver_rewards.append(aver_rewards)
        #
        self.log_info("finished")
        #
        return self.list_aver_rewards
        #
