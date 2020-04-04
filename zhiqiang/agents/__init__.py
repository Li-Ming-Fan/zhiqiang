

from abc import ABCMeta, abstractmethod

from zhiqiang.utils import torch_utils

#
class AbstractAgent(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    #
    @abstractmethod
    def act(self, observation):
        """ choose an action, based on observation
            return: action
        """
        pass

    @abstractmethod
    def generate(self, max_gen_step, env, observation=None):
        """ return list_experiences
        """
        pass
    #

    #
    @abstractmethod
    def standardize_batch(self, batch_data):
        """ batch_data: dict, {"data": data, "position": posi}
            return: batch_std
        """
        pass

    @abstractmethod
    def optimize(self, batch_std, buffer=None):
        """ optimization step            
            buffer: replay_buffer, for possible update
        """
        pass
    
    @abstractmethod
    def update_base_net(self):
        """
        """
        pass
    #

    #
    @abstractmethod
    def train_mode(self):
        """
        """
        pass

    @abstractmethod
    def eval_mode(self):
        """
        """
        pass
    #

    #
    def rollout(self, max_step, env, observation=None):
        """
        """
        sum_rewards = 0
        list_transitions = []
        #
        if observation is None:
            observation = env.reset()
        #
        for step in range(max_step):
            action = self.act(observation)
            sp, reward, done, info_env = env.step(action)
            exp = (observation, action, reward, sp, {"info_env": info_env})
            observation = sp
            #
            sum_rewards += reward
            list_transitions.append(exp)
            #
            if done: break
            #
        #
        return sum_rewards, list_transitions
        #

    def eval(self, num_rollout, max_step, env, observation=None):
        """
        """
        aver_rewards = 0
        for idx in range(num_rollout):
            rewards, list_transitions = self.rollout(max_step, env)
            aver_rewards += rewards
        #
        return aver_rewards / num_rollout
        #
    #

#
class AbstractPQNet(metaclass=ABCMeta):
    """
    """
    def __init__(self):
        """
        """
        pass

    @abstractmethod
    def trans_list_observations(self, list_observations):
        """ trans list_observations to batch_std for model
            return: batch_std, dict
        """
        pass

    @abstractmethod
    def infer(self, batch_std):
        """ return: action_values, or policy
        """
        pass

    #
    def eval_mode(self):
        self.eval()

    def train_mode(self):
        self.train()

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def back_propagate(self, loss):
        loss.backward(retain_graph=False)
        self.optimizer.step()
    #
    def merge_weights_function(self):
        return torch_utils.merge_weights
    #
    
    

