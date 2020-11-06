
import os
import time
import numpy as np
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import imageio


from zhiqiang.envs import AbstractEnv


def int_round(value):
    return int(round(value, 0))


class Pong(AbstractEnv):
    """
    """
    def __init__(self, settings):
        """
        """
        self.settings = settings
        self.sizeH = self.settings.env_settings["size_h"]
        self.sizeW = self.settings.env_settings["size_w"]
        self.ball_v = self.settings.env_settings["ball_v"]
        self.max_v = self.settings.env_settings["max_v"]
        #
        self.num_players = 1
        self.reward_hit = 1
        self.reward_loss = -1
        #
        self.reset()        
        #

    def reset(self):
        """
        """
        self._reset_env()
        return self._reset_game()

    #
    def _reset_env(self):
        """
        """
        self.list_actions = list(range(-self.max_v, self.max_v+1))
        self.num_actions = len(self.list_actions)
        #
        self.racket_h = 3
        self.racket_w = 1
        self.ball_size = 1
        #
        # board
        self.size_all_h = self.sizeH + 2
        self.size_all_w = self.sizeW + 2
        self.pa_h = self.size_all_h - 1   # all
        self.pa_w = self.size_all_w - 1
        self.pd_h = self.size_all_h - 2   # board
        self.pd_w = self.size_all_w - 2
        #
        self.size_center_h = self.size_all_h // 2
        self.size_center_w = self.size_all_w // 2
        #
        self.flag_reset_game = 1
        #
        self.pic_h = self.size_all_h
        self.pic_w = self.size_all_w
        #

    def _reset_game(self):
        """
        """
        # start, position, player, ball
        self.pp_0 = self.size_center_h    # player
        self.pp_1 = self.size_center_h
        self.pb_h = self.size_center_h    # ball
        self.pb_w = self.size_center_w
        #
        self.mb_h = np.random.randint(-10, 10)  # movement, ball
        self.mb_w = np.random.randint(2, 10)
        self._normalize_mb()
        #
        self.action_player_0 = 0
        self.action_player_1 = 0
        #
        self.flag_reset_game = 0
        #
        # state
        self.state_prev = self.render()
        self.state_curr = self.render()
        self.state = (self.state_prev, self.state_curr)
        return self.state
        #
    
    def render(self):
        """
        """
        a = np.ones([self.size_all_h, self.size_all_w, 3])
        a[1:-1,1:-1, 0] = 0
        a[1:-1,1:-1, 1] = 1
        a[1:-1,1:-1, 2] = 1
        #
        if self.num_players == 1:
            pass
        else:
            a[self.pp_0-1:self.pp_0+2, 0, 0] = 1
            a[self.pp_0-1:self.pp_0+2, 0, 1] = 0
            a[self.pp_0-1:self.pp_0+2, 0, 2] = 0
        #
        a[self.pp_1-1:self.pp_1+2, self.pa_w, 0] = 0
        a[self.pp_1-1:self.pp_1+2, self.pa_w, 1] = 0
        a[self.pp_1-1:self.pp_1+2, self.pa_w, 2] = 1
        #
        a[self.pb_h, self.pb_w, :] = 0           
        #
        return a
        #
    
    def step(self, action, action_0=0):
        """
        """
        self.action_player_0 = self.list_actions[ action_0 ]
        self.action_player_1 = self.list_actions[ action ]
        #
        if self.flag_reset_game:
            return self._reset_game(), self._score(0.0, 0.0), False, {"reset": 1}
        #
        # ball
        d_h = self.mb_h
        d_w = self.mb_w
        #
        # check, reward
        #
        ## first
        flag_reset_game, reward_0, reward_1 = self._check_bound_w(d_h, d_w)
        #
        # reset
        if flag_reset_game:
            #
            self.flag_reset_game = 1
            #
            self.state_prev = self.state_curr
            self.state_curr = self.render()
            self.state = (self.state_prev, self.state_curr)
            #
            return self.state, self._score(reward_0, reward_1), False, None
            #
        #
        self._check_bound_h(d_h, d_w)
        #

        #
        # player
        self._execute_player_action(0)
        self._execute_player_action(1)
        #
        # ball_v
        self._normalize_mb()
        #
        
        #
        self.state_prev = self.state_curr
        self.state_curr = self.render()
        self.state = (self.state_prev, self.state_curr)
        #
        return self.state, self._score(reward_0, reward_1), False, None
        #
    
    def close(self):
        pass
    #
    
    #
    def map_to_pic(self, state_all):
        """
        """
        state = state_all[1]
        #
        h = self.pic_h
        w = self.pic_w
        b = scipy.misc.imresize(state[:,:,0],[h,w,1],interp='nearest')
        c = scipy.misc.imresize(state[:,:,1],[h,w,1],interp='nearest')
        d = scipy.misc.imresize(state[:,:,2],[h,w,1],interp='nearest')
        pic = np.stack([b,c,d], axis=2)
        #        
        return pic

    def display(self, state=None, show=True, step=None, score=None):
        """
        """
        if state is None:
            state = self.state
        #
        # grid = plt.GridSpec(1, 5, wspace=0.5, hspace=0.5)
        # plt.subplot(grid[0, 0:3])
        pic = self.map_to_pic(state)
        plt.imshow(pic, interpolation="nearest")
        # plt.subplot(grid[0, 4])
        if step is not None:
            if score is not None:
                plt.title("step: %d, score: %f" % (step, score))
            else:
                plt.title("step: %d" % (step, ))
        else:
            if score is not None:
                plt.title("score: %f" % (score, ))
            #
        #
        if show:
            plt.show()
        #
        filename = "pong_temp.eps"  # eps < png < jpg
        plt.savefig(filename)
        pic_with_score = imageio.imread(filename) 
        os.remove(filename)
        #
        return pic, pic_with_score
        #
    #

    #
    def _execute_player_action(self, player_idx):
        """
        """
        if player_idx == 0:
            self.pp_0 += self.action_player_0
            # 
            if self.pp_0 < 1:
                self.pp_0 = 1
            elif self.pp_0 > self.pd_h:   # board
                self.pp_0 = self.pd_h
            #
        elif player_idx == 1:
            self.pp_1 += self.action_player_1
            #
            if self.pp_1 < 1:
                self.pp_1 = 1
            elif self.pp_1 > self.pd_h:
                self.pp_1 = self.pd_h
            #
    #
    def _normalize_mb(self):
        """
        """
        norm = np.sqrt(self.mb_h ** 2 + self.mb_w ** 2)
        ratio = self.ball_v / norm
        self.mb_h = int_round(self.mb_h * ratio)
        self.mb_w = int_round(self.mb_w * ratio)
    #
    def _check_bound_w(self, d_h, d_w):
        """
        """
        # check, reward
        flag_reset_game = 0
        reward_0 = 0
        reward_1 = 0
        #
        if self.pb_w + d_w <= 0:
            #
            if self.num_players == 1:
                # reflection
                self.pb_w += d_w
                self.pb_w = 1 - self.pb_w
                self.mb_w = -self.mb_w
                #
            else:
                #
                # when pb_w = 1,
                d_pbw = 1 - self.pb_w
                d_pbh = int_round(d_h * d_pbw / d_w)
                d_p0 = int_round(self.action_player_0 * d_pbw / d_w)
                #
                pb_h = self.pb_h + d_pbh
                p0 = self.pp_0 + d_p0
                #
                if pb_h <= 0:
                    pb_h = 1 - pb_h
                elif pb_h >= self.pa_h:
                    pb_h = 2 * self.pd_h - pb_h - 1
                #
                if p0 <= 0: 
                    p0 = 1
                elif p0 >= self.pa_h:
                    p0 = self.pd_h
                #
                if pb_h >= p0-1 and pb_h <= p0+1:
                    # reflection
                    self.pb_w += d_w
                    self.pb_w = 1 - self.pb_w
                    self.mb_w = -self.mb_w
                    #
                    reward_0 = self.reward_hit
                    #
                else:
                    reward_0 = self.reward_loss
                    #
                    flag_reset_game = 1
                    self.pb_w = 0
                    self.pb_h = pb_h
                    self.pp_0 = p0
                    self._execute_player_action(1)
                    #
            #
        elif self.pb_w + d_w >= self.pa_w:
            #
            # when pb_w = self.pd_w,
            d_pbw = self.pd_w - self.pb_w
            d_pbh = int_round(d_h * d_pbw / d_w)
            d_p1 = int_round(self.action_player_1 * d_pbw / d_w)
            #
            pb_h = self.pb_h + d_pbh
            p1 = self.pp_1 + d_p1
            #
            if pb_h <= 0:
                pb_h = 1 - pb_h
            elif pb_h >= self.pa_h:
                pb_h = 2 * self.pd_h - pb_h - 1
            #
            if p1 <= 0: 
                p1 = 1
            elif p1 >= self.pa_h:
                p1 = self.pd_h
            #
            if pb_h >= p1-1 and pb_h <= p1+1:
                # reflection
                self.pb_w += d_w
                self.pb_w = 2 * self.pd_w - self.pb_w + 1
                self.mb_w = -self.mb_w
                #
                reward_1 = self.reward_hit
                #
            else:
                reward_1 = self.reward_loss
                #
                flag_reset_game = 1
                self.pb_w = self.pa_w
                self.pb_h = pb_h
                self.pp_1 = p1
                self._execute_player_action(0)
                #
            #
        else:
            self.pb_w += d_w
            #
        #
        return flag_reset_game, reward_0, reward_1
        #
    #
    def _check_bound_h(self, d_h, d_w):
        """
        """
        # check, reflection, h
        self.pb_h += d_h
        #
        if self.pb_h <= 0:
            self.pb_h = 1 - self.pb_h
            self.mb_h = -self.mb_h -1
        elif self.pb_h >= self.pa_h:
            self.pb_h = 2 * self.pd_h - self.pb_h - 1
            self.mb_h = -self.mb_h +1
        #
    #
    def _score(self, reward_0, reward_1):
        """
        """
        if self.num_players == 1:
            return reward_1
        else:
            return np.array( [reward_0, reward_1] )
        #

#
