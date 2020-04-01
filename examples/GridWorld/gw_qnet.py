

from zhiqiang.agents import AbstractQNet


class GridWorldQNet(AbstractQNet):
    """
    """
    def __init__(self, agent_settings):
        """
        """
        pass

    def trans_list_observations(self, list_observation):
        """ trans list_observation to batch_std for model
            return: batch_std, dict
        """
        pass

    def infer(self, observation):
        """
        """
        """
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.sizeInput = world.state.size
        self.scalarInput = tf.placeholder(shape = [None, world.state.size],\
                                          dtype = tf.float32)
        #
        # feature extraction
        s = world.state.shape;
        shape = [-1, s[0], s[1], s[2]];
        self.imageIn = tf.reshape(self.scalarInput,shape=shape)
        # 7*7 --> 6*6 --> 5*5 --> 3*3 --> 1*1
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,\
            kernel_size=[2,2], stride=[1,1], padding='VALID',biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,\
            kernel_size=[2,2], stride=[1,1], padding='VALID',biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,\
            kernel_size=[3,3], stride=[1,1], padding='VALID',biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3,num_outputs=n_feat,\
            kernel_size=[3,3], stride=[1,1], padding='VALID',biases_initializer=None)
        #
        """

    def back_propagate(self, loss):
        """
        """
        pass

    def merge_weights(self, another_qnet, merge_ksi):
        """
        """
        pass


