
import os
import argparse

from zhiqiang.utils.basic_settings import BasicSettings

import torch


env = "GridWorld"
agent = "VanilaDQN"
buffer = "SimpleBuffer"
trainer = "SimpleTrainer"

mode = "train"
seed = 100

data_root = "./data_root"
dir_rel_log = "log"
dir_rel_settings = "settings"
dir_rel_model = "stat_dict"
settings_file = "settings_dqn.json"


def parsed_args():
    """
    """
    # Hyper Parameters
    parser = argparse.ArgumentParser()    
    parser.add_argument('--env', default=env, type=str)
    parser.add_argument('--agent', default=agent, type=str)
    parser.add_argument('--buffer', default=buffer, type=str)
    parser.add_argument('--trainer', default=trainer, type=str)
    #
    parser.add_argument('--mode', default=mode, type=str, help='train, eval, play')
    #
    parser.add_argument('--data_root', default=data_root, type=str)
    parser.add_argument('--dir_rel_log', default=dir_rel_log, type=str)
    parser.add_argument('--dir_rel_settings', default=dir_rel_settings, type=str)
    parser.add_argument('--dir_rel_model', default=dir_rel_model, type=str)
    parser.add_argument('--settings_file', default=settings_file, type=str)
    #
    parser.add_argument('--num_play_times', default=5, type=int)
    #
    parser.add_argument('--save', default=1, type=int)               # bool
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--device_type', default=None, type=str)
    #
    settings = parser.parse_args()
    return settings

#
def settings_with_args(args):
    """
    """
    settings_filepath = os.path.join(args.data_root, args.dir_rel_settings,
                                     args.settings_file)
    settings = BasicSettings(settings_filepath)
    settings.assign_info_from_namedspace(args)
    #
    return settings

#
def main(settings):
    """
    """
    from atari_env import atari_env_list
    # env
    if settings.env == "GridWorld":
        from grid_world import GridWorld as Env
        from gridworld_pnet import GridWorldPNet as PNet
        from gridworld_qnet import GridWorldQNet as QNet
        from gridworld_vnet import GridWorldVNet as VNet
        agent_modules = {"qnet": QNet, "pnet": PNet, "vnet": VNet}
    elif settings.env in atari_env_list:
        from atari_env import get_atari_env
        Env = get_atari_env(settings.env, settings)
        from atari_qnet import AtariQNet as QNet
        from atari_pnet import AtariPNet as PNet
        from atari_vnet import AtariVNet as VNet
        agent_modules = {"qnet": QNet, "pnet": PNet, "vnet": VNet}

    # agent
    if settings.agent == "VanilaDQN":
        from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
    elif settings.agent == "DoubleDQN":
        from zhiqiang.agents.dqn_double import DoubleDQN as Agent
    elif settings.agent == "MStepDQN":
        from zhiqiang.agents.dqn_mstep import MStepDQN as Agent
    elif settings.agent == "PriorityDQN":
        from zhiqiang.agents.dqn_priority import PriorityDQN as Agent
    elif settings.agent == "EntropyACQ":
        from zhiqiang.agents.acq_entropy import EntropyACQ as Agent
    elif settings.agent == "EntropyACV":
        from zhiqiang.agents.acv_entropy import EntropyACV as Agent
    elif settings.agent == "MStepPolicy":
        from zhiqiang.agents.policy_mstep import MStepPolicy as Agent
    
    # buffer
    if settings.buffer == "SimpleBuffer":
        from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
    elif settings.buffer == "PriorityBuffer":
        from zhiqiang.replay_buffers.priority_buffer import PriorityBuffer as Buffer
        
    # trainer
    if settings.trainer == "SimpleTrainer":
        from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
    elif settings.trainer == "ParalTrainer":
        from zhiqiang.trainers.paral_trainer import ParalTrainer as Trainer
    #

    # device
    settings.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if settings.device_type is None else torch.device(settings.device_type)
    #

    # trainer_instance
    trainer_inst = Trainer(settings, Agent, agent_modules, Env, Buffer)
    #
    if settings.mode == "train":
        list_aver_rewards = trainer_inst.do_train()
    elif settings.mode == "eval":
        aver_rewards = trainer_inst.do_eval(trainer_inst.num_eval_rollout)
    elif settings.mode == "play":
        list_rewards = trainer_inst.do_play(settings.num_play_times)
    #

    # close logger
    settings.close_logger()
    #

#
if __name__ == '__main__':
    """
    """
    args = parsed_args()
    settings = settings_with_args(args)
    main(settings)



