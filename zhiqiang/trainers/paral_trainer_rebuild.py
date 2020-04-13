

from . import AbstractTrainer
from ..utils.data_parallelism import DataParallelism

import os
import logging

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.multiprocessing as mp


#
def process_eval(list_data, idx, result_dict, settings_paral):
    """
    """
    max_step = settings_paral["max_step"]
    agent = settings_paral["list_workers"][idx]
    #
    list_result = []
    for item in list_data:
        list_rewards, list_trans = agent.rollout(max_step)
        list_result.append(sum(list_rewards) )
    #
    result_dict[idx] = list_result
    #

def process_gen(list_data, idx, result_dict, settings_paral):
    """
    """
    max_step = settings_paral["max_step"]
    base_rewards = settings_paral["max_aver_rewards"]
    agent = settings_paral["list_workers"][idx]
    #
    list_result = []
    for item in list_data:
        list_expr = agent.generate(base_rewards, max_step)
        if len(list_expr) > 0:          # list of experiences, training sample
            list_result.append(list_expr)     
    #
    result_dict[idx] = list_result
    #

def merge_eval_result(result_dict, settings_paral):
    """
    """
    num_workers = settings_paral["num_workers"]
    #
    sum_rewards = 0
    for idx in range(num_workers):
        sum_curr = sum(result_dict[idx])
        sum_rewards += sum_curr
    #
    return sum_rewards

def merge_gen_result(result_dict, settings_paral):
    """
    """
    num_workers = settings_paral["num_workers"]
    #
    list_all = []
    for idx in range(num_workers):
        list_all.extend(result_dict[idx])
    #
    return list_all         # list of experiences
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
        self.agent = agent_class(settings, agent_modules)
        self.agent.env = env_class(settings)
        self.agent.to(settings.device)
        self.agent.train_mode()                  # train mode
        # learner agent (optimization)

        self.buffer = buffer_class(settings)

        #
        # self.logger = settings.create_logger(settings.log_path)
        self.logger = logging.getLogger(settings.log_path)
        settings.logger = None               # logger, cannot be pickled
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

        # worker agent (evaluation, exploration)
        self.num_workers = self.settings.trainer_settings["num_workers"]
        #
        self.list_workers = []
        for idx in range(self.num_workers):
            agent = agent_class(settings, agent_modules)
            agent.env = env_class(settings)
            # agent.to(self.settings.device)
            self.list_workers.append(agent)
        #
        # data_paral
        self.settings_paral = {"num_workers": self.num_workers }
        self.settings_paral["max_aver_rewards"] = self.max_aver_rewards
        self.settings_paral["max_step"] = self.max_step
        self.settings_paral["list_workers"] = self.list_workers
        #
        self.data_paral_eval = DataParallelism(self.num_workers,
                    process_eval, merge_eval_result, self.settings_paral)
        #
        self.data_paral_gen = DataParallelism(self.num_workers,
                    process_gen, merge_gen_result, self.settings_paral)
        #
        # import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        #

    def log_info(self, str_info):
        """
        """
        print(str_info)
        self.logger.info(str_info)

    #
    def set_workers_mode(self, mode):
        """
        """
        self.settings_paral = {"num_workers": self.num_workers }
        self.settings_paral["max_aver_rewards"] = self.max_aver_rewards
        self.settings_paral["max_step"] = self.max_step
        #
        for idx in range(self.num_workers):
            if mode == "eval":
                self.list_workers[idx].eval_mode()
            elif mode == "explore":
                self.list_workers[idx].explore_mode()
            #
        #
        self.settings_paral["list_workers"] = self.list_workers
        #

    def update_workers_params(self):
        """
        """
        for idx in range(self.num_workers):
            self.list_workers[idx].copy_params(self.agent)
        #

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
        self.set_workers_mode("eval")                # eval mode
        self.data_paral_eval.do_processing(list(range(num_rollout)), rebuild=True)
        #
        merged_result = self.data_paral_eval.merged_result
        aver_rewards = merged_result / num_rollout
        #
        self.data_paral_eval.clear_result_dict()
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
        self.set_workers_mode("explore")               # explore mode
        self.data_paral_gen.do_processing(list(range(num_rollout)), rebuild=True)
        #
        merged_result = self.data_paral_gen.merged_result
        #
        for item in merged_result:     # list of experiences
            self.buffer.add(item)
        #
        count_better = len(merged_result)
        #
        self.data_paral_gen.clear_result_dict()
        #
        self.log_info("count_better: %d" % count_better)
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
