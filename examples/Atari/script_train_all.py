
from zargs_trainer import parsed_args, settings_with_args, main

import copy

if __name__ == "__main__":

    args = parsed_args()

    ##
    args.env = "Pong-v0"
    args.settings_file = "settings_atari.json"


    ##
    args.trainer = "SimpleTrainer"
    # args.trainer = "ParalTrainer"
    settings_base = settings_with_args(args)

    ##
    agent_list = ["VanilaDQN", "DoubleDQN", "MStepDQN", "PriorityDQN" ]
    agent_list += ["EntropyACQ", "EntropyACV", "MStepPolicy" ]
    #
    agent_list = ["VanilaDQN" ]
    #

    for agent in agent_list:
        #
        settings = copy.deepcopy(settings_base)
        settings.agent = agent
        
        if "Priority" in agent:
            settings.buffer = "PriorityBuffer"
        else:
            settings.buffer = "SimpleBuffer"
        
        #
        settings.display()
        main(settings)
        #
    #

    


