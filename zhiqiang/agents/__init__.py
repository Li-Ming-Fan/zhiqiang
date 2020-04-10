

#
class AbstractAgent(object):
    """
    necessary_elements:
        function act(self, observation),
        choose an action, based on observation,
        for play, exploration,
        return: action

        function act_with_learner(self, observation),
        choose an action, based on observation,
        for evaluation of the learner
        return: action

        function generate(self, base_rewards, max_gen_step, observation=None)
        generate experiences by a rollout
        returning: list_experiences

        ---

        function standardize_batch(self, batch_data),
        batch_data: dict, {"data": data, "position": posi}
        return: batch_std

        function optimize(self, batch_std, buffer=None),
        optimization step            
        buffer: replay_buffer, for possible update
        returning nothing

        function update_base_net(self),
        returning nothing

        function copy_params(self, another),
        returning nothing

        ---

        function train_mode(self),
        same functionality with model.train() in torch,
        but can be defined with more operations
        returning nothing

        function eval_mode(self),
        same functionality with model.eval() in torch,
        but can be defined with more operations
        returning nothing

        function explore_mode(self),
        returning nothing

        ---

        function load(self, model_path),
        returning nothing

        function save(self, model_path),
        returning nothing

    implemented_elements:
        function rollout(self, max_step, observation=None, mode="learner"),
        returning list_reward, list_transitions

        function eval(self, num_rollout, max_step, observation=None, mode="learner"),
        returning sum_total_rewards / num_rollout
    """
    necessary_elements = ["act", "act_with_learner", "generate"]
    necessary_elements += ["standardize_batch", "optimize"]
    necessary_elements += ["update_base_net", "copy_params"]
    necessary_elements += ["train_mode", "eval_mode", "explore_mode"]
    necessary_elements += ["load", "save"]
    #
    def print_info():
        print("-" * 70)
        print(AbstractAgent.__doc__)
        print("-" * 70)
    #
    def check_necessary_elements(self, subclass_name):
        """
        """
        subclass_elements = subclass_name.__dict__
        for item in AbstractAgent.necessary_elements:
            if item not in subclass_elements:
                AbstractAgent.print_info()
                assert False, "function %s NOT implemented. See above for information." % item 
    #
    def __init__(self):
        pass
    #
    def rollout(self, max_step, observation=None, mode="learner"):
        """
        """
        list_reward = []
        list_transitions = []
        #
        if observation is None:
            observation = self.env.reset()
        #
        if mode == "learner":
            act_function = self.act_with_learner   # for eval
        else:   # "base", or others
            act_function = self.act  # for exploration, play
        #
        for step in range(max_step):
            action = act_function(observation)
            sp, reward, done, info_env = self.env.step(action)
            exp = (observation, action, reward, sp, {"info_env": info_env})
            observation = sp
            #
            list_reward.append(reward)
            list_transitions.append(exp)
            #
            if done: break
            #
        #
        return list_reward, list_transitions
        #

    def eval(self, num_rollout, max_step, observation=None, mode="learner"):
        """
        """
        sum_total_rewards = 0
        for idx in range(num_rollout):
            list_r, list_transitions = self.rollout(max_step, observation, mode)
            sum_total_rewards += sum(list_r)
        #
        return sum_total_rewards / num_rollout
        #
    #

#
class AbstractPQNet(object):
    """
    necessary_elements:
        function trans_list_observations(self, list_observations),
        trans list_observations to batch_std for model
        returning: batch_std, dict

        function infer(self, batch_std),
        returning: action_values, or policy, or other

        function merge_weights_function(self),
        returnining: a function, such as torch_utils.merge_weights
    
    implemented_elements:
        function eval_mode(self),
        returnining nothing

        function train_mode(self),
        returnining nothing

        function back_propagate(self, loss),
        returnining nothing
    """
    necessary_elements = ["trans_list_observations", "infer"] 
    necessary_elements += ["merge_weights_function"]
    #
    def print_info():
        print("-" * 70)
        print(AbstractPQNet.__doc__)
        print("-" * 70)
    #
    def check_necessary_elements(self, subclass_name):
        """
        """
        subclass_elements = subclass_name.__dict__
        for item in AbstractPQNet.necessary_elements:
            if item not in subclass_elements:
                AbstractPQNet.print_info()
                assert False, "function %s NOT implemented. See above for information." % item 
    #
    def __init__(self):
        pass
    #
    """
    Assuming torch. They can be override to become customized.
    """ 
    #
    def eval_mode(self):
        self.eval()

    def train_mode(self):
        self.train()
        
    def back_propagate(self, loss):
        self.optimizer.zero_grad()       
        loss.backward(retain_graph=False)
        self.optimizer.step()
    #
    
    
#
if __name__ == "__main__":

    AbstractPQNet.print_info()
    AbstractAgent.print_info()
