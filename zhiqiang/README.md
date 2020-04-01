# zhiqiang

zhiqiang, 之强, become strong. Similar to ziqiang, 自强, Self-strengthening.

A package for reinforcement learning algorithms. PyTorch / Tensorflow.


## Description

Abstract classes that form the framework:
```
from zhiqiang.agents import AbstractAgent
from zhiqiang.envs import AbstractEnv
from zhiqiang.replay_buffers import AbstractBuffer
from zhiqiang.trainers import AbstractTrainer
```

Implemented Trainers and Buffers:
```
from zhiqiang.trainers.simple_trainer import SimpleTrainer as Trainer
from zhiqiang.replay_buffers.simple_buffer import SimpleBuffer as Buffer
from zhiqiang.replay_buffers.priority_buffer import PriorityBuffer as Buffer
```

Implemented Agents:
```
from zhiqiang.agents.dqn_vanila import VanilaDQN as Agent
from zhiqiang.agents.dqn_double import DoubleDQN as Agent

```


## Installation

From PyPI distribution system:

```
pip install zhiqiang
```


## Usage

For usage examples of this package, please see:

1, examples/

2, TODO


</br>
