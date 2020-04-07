

#
class AbstractAgent(object):
    """
    """
    str_sep = "-"*70
    necessary_elements = ["act", "generate"]
    necessary_elements += ["standardize_batch", "optimize", "update_base_net"]
    necessary_elements += ["train_mode", "eval_mode", "explore_mode"]
    #
    necessary_elements_info = """\n%s
    necessary_elements:
    >   function act(self, observation),
        choose an action, based on observation
        return: action

        function generate(self, base_rewards, max_gen_step, env, observation=None)
        generate experiences by a rollout
        returning: list_experiences

        ---

        function standardize_batch(self, batch_data),
        batch_data: dict, {"data": data, "position": posi}
        return: batch_std

        function optimize(self, batch_std, buffer=None),
        optimization step            
        buffer: replay_buffer, for possible update
        returnining nothing

        function update_base_net(self),
        returnining nothing

        ---

        function train_mode(self),
        same functionality with model.train() in torch,
        but can be defined with more operations
        returnining nothing

        function eval_mode(self),
        same functionality with model.eval() in torch,
        but can be defined with more operations
        returnining nothing

        function explore_mode(self),
        returnining nothing
    \n
    implemented_elements:
    >   function rollout(self, max_step, env, observation=None),
        returnining total_rewards, list_transitions

        function eval(self, num_rollout, max_step, env, observation=None),
        returnining sum_total_rewards / num_rollout
    \n%s
    """ % (str_sep, str_sep)
    #
    def print_info():
        print(AbstractAgent.necessary_elements_info)
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
    def rollout(self, max_step, env, observation=None):
        """
        """
        total_rewards = 0
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
            total_rewards += reward
            list_transitions.append(exp)
            #
            if done: break
            #
        #
        return total_rewards, list_transitions
        #

    def eval(self, num_rollout, max_step, env, observation=None):
        """
        """
        sum_total_rewards = 0
        for idx in range(num_rollout):
            rewards, list_transitions = self.rollout(max_step, env)
            sum_total_rewards += rewards
        #
        return sum_total_rewards / num_rollout
        #
    #

#
class AbstractPQNet(object):
    """
    """
    str_sep = "-"*70
    necessary_elements = ["trans_list_observations", "infer"] 
    necessary_elements += ["merge_weights_function"]
    #
    necessary_elements_info = """\n%s
    necessary_elements:
    >   function trans_list_observations(self, list_observations),
        trans list_observations to batch_std for model
        returning: batch_std, dict

        function infer(self, batch_std),
        returning: action_values, or policy, or other

        function merge_weights_function(self),
        returnining: a function, such as torch_utils.merge_weights
    \n
    implemented_elements:
    >   function eval_mode(self),
        returnining nothing

        function train_mode(self),
        returnining nothing

        function back_propagate(self, loss),
        returnining nothing
    \n%s
    """ % (str_sep, str_sep)
    #
    def print_info():
        print(AbstractPQNet.necessary_elements_info)
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
    Assuming torch
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
    
    

