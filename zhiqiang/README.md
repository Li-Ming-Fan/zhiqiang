# zhiqiang

zhiqiang, 之强, become strong. And similar to ziqiang, 自强, Self-strengthening.

A platform for reinforcement learning algorithms.

Work with PyTorch, but only in the implemented concrete agents and utils.torch_utils.


## Examples

Learning curriculum of different agents for the environment GridWorld:

![learning_curriculum](url_link)


A replay of a trained VanilaDQN:

![a_replay_gif](url_link)


## Description

Abstract classes that form the framework:
```
from zhiqiang.agents import AbstractPQNet
from zhiqiang.agents import AbstractAgent
from zhiqiang.envs import AbstractEnv
from zhiqiang.replay_buffers import AbstractBuffer
from zhiqiang.trainers import AbstractTrainer
```

Run commands such as
```
AbstractPQNet.print_info()
AbstractAgent.print_info()
```
to see necessary functions for implementing concrete classes.


Implemented Trainers and Buffers:
```
from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
from zhiqiang.trainers.paral_trainer import ParalTrainer as Trainer
from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
from zhiqiang.replay_buffers.priority_buffer import PriorityBuffer as Buffer
```

Implemented Agents:
```
from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
from zhiqiang.agents.dqn_double import DoubleDQN as Agent
from zhiqiang.agents.dqn_mstep import MStepDQN as Agent
from zhiqiang.agents.dqn_priority import PriorityDQN as Agent
```

More:
```
.
├── __init__.py
├── agents
│   ├── __init__.py
│   ├── acq_entropy.py
│   ├── acv_entropy.py
│   ├── dqn_double.py
│   ├── dqn_mstep.py
│   ├── dqn_priority.py
│   ├── dqn_vanila.py
│   └── policy_mstep.py
├── envs
│   └── __init__.py
├── replay_buffers
│   ├── __init__.py
│   ├── priority_buffer.py
│   └── simple_buffer.py
├── trainers
│   ├── __init__.py
│   ├── paral_trainer.py
│   └── simple_trainer.py
└── utils
    ├── __init__.py
    ├── basic_settings.py
    ├── data_parallelism.py
    ├── log_parser.py
    ├── torch_utils.py
    └── uct_simple.py
```

## Quick Trial

For a quick trial, please try codes in the file examples/GridWorld/script_train_simple.py:

```
# define an env
from grid_world import GridWorld as Env

# define a qnet, in PyTorch
from gridworld_qnet import GridWorldQNet as QNet

# pick an agent
from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
# from zhiqiang.agents.dqn_double import DoubleDQN as Agent
# from zhiqiang.agents.dqn_mstep import MStepDQN as Agent
# from zhiqiang.agents.dqn_priority import PriorityDQN as Agent


# pick a buffer
from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
# from zhiqiang.replay_buffers.priority_buffer import PriorityBuffer as Buffer

# pick a trainer
from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
# from zhiqiang.trainers.paral_trainer import ParalTrainer as Trainer

# settings file
settings_filepath = "./examples/GridWorld/settings_gridworld.json"

##
#
from zhiqiang.utils.basic_settings import BasicSettings
#
settings = BasicSettings(settings_filepath)
settings.check_settings()
settings.display()
#
trainer = Trainer(settings, Agent, {"qnet": QNet}, Env, Buffer)
#
# train
list_aver_rewards = trainer.do_train()
#
# draw
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 5))
#
eval_period = settings.trainer_settings["eval_period"]
list_x = [idx * eval_period for idx in range(len(list_aver_rewards))]
#
print(list_x)
print(list_aver_rewards)
#
plt.plot(list_x, list_aver_rewards, label="Averaged Rewards", color="r", linewidth=2)
plt.xlabel("Number Boost")
plt.ylabel("Averaged Rewards")  # plt.title("Boost Curriculum")
# plt.xticks(list_x)              # plt.legend()
plt.grid()
plt.show()
```

For utilizing more agents, please see codes in the file examples/GridWorld/script_train_all.py. 


## Installation

From PyPI distribution system:

```
pip install zhiqiang
```

## Usage

For usage examples of this package, please see:

1, examples/

</br>
