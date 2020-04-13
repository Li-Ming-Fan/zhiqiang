# ZhiQiang, 之强

zhiqiang, 之强, become strong. And similar to ziqiang, 自强, Self-strengthening.

A platform for reinforcement learning. The framework does not depend on any specific deep learning platform. But the implemented concrete agents are written with PyTorch.


## Examples

Learning curriculum of different agents for the environment GridWorld:

<img src="https://github.com/Li-Ming-Fan/zhiqiang/blob/master/aaa_store/learning_curriculum.png" width="50%" height="50%" alt="learning_curriculum">


A replay of a trained EntropyACV agent for GridWorld:

<img src="https://github.com/Li-Ming-Fan/zhiqiang/blob/master/aaa_store/a_replay_gif.gif" width="30%" height="30%" alt="gridworld_replay_gif">


## Description

Abstract classes that form the framework:
```
from zhiqiang.agents import AbstractPQNet
from zhiqiang.agents import AbstractAgent
from zhiqiang.envs import AbstractEnv
from zhiqiang.replay_buffers import AbstractBuffer
from zhiqiang.trainers import AbstractTrainer
```

Please run commands such as
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

Some of the implemented agents:
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

# settings file, make sure the path is right
settings_filepath = "./data_root/settings/settings_gridworld.json"
agent_name = "agentname"
env_name = "GridWorld"

##
#
from zhiqiang.utils.basic_settings import BasicSettings
#
settings = BasicSettings(settings_filepath)
settings.env = env_name
settings.agent = agent_name
settings.check_settings()
settings.display()
#
# device
import torch
settings.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
    if settings.device_type is None else torch.device(settings.device_type)
#
print("device: {}".format(settings.device))
#
# trainer
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
plt.ylabel("Averaged Rewards")    # plt.title("Boost Curriculum")
# plt.xticks(list_x)              # plt.legend()
plt.grid()
plt.show()
```

For utilization of more agents, please see codes in the file examples/GridWorld/script_train_all.py.


## Philosophy

This package does not aim to encompass all kinds of reinforcement learning algorithms, but just to provide a framework for RL solutions of tasks.

An RL solution always involves an environment, an agent (agents) and some neural networks (as agent modules). For training the agent (agents), a trainer and a replay buffer are further required. If interface functions among these parts are well defined, then the different parts can be easy to change as plug-and-play. This is what this package aims to do.

In this package, a set of inferface functions is defined, and some simple implementations of the different parts are conducted. We hope these will pave way for users to make their own customized definitions and implementations. 


## Installation

From PyPI distribution system:

```
pip install zhiqiang
```

This package is tested with PyTorch 1.4.0.


## Usage

For usage examples of this package, please see:

1, examples/GridWorld

2, examples/Atari


## Citation

If you find ZhiQiang helpful, please cite it in your publications.

```
@software{zhiqiang,
  author = {Ming-Fan Li},
  title = {ZhiQiang, a platform for reinforcement learning},
  year = {2020},
  url = {https://github.com/Li-Ming-Fan/zhiqiang}
}
```


</br>

