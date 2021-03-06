
import os
import time
import numpy as np
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import imageio

from zhiqiang.envs import AbstractEnv


class GridItem():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class GridWorld(AbstractEnv):
    """
    """
    def __init__(self, settings):
        """
        """
        self.settings = settings
        self.sizeX = self.settings.env_settings["size"]
        self.sizeY = self.settings.env_settings["size"]
        self.actions = 4
        self.objects = []
        self.partial = self.settings.env_settings["partial"]
        #
        self.reset()        
        #
        
    def reset(self):
        """
        """
        self.objects = []
        #
        hero = GridItem(self.get_new_posi(),1,1,2,None,'hero')
        self.objects.append(hero)
        #
        food = GridItem(self.get_new_posi(),1,1,1,1,'food')
        self.objects.append(food)
        food2 = GridItem(self.get_new_posi(),1,1,1,1,'food')
        self.objects.append(food2)
        food3 = GridItem(self.get_new_posi(),1,1,1,1,'food')
        self.objects.append(food3)
        food4 = GridItem(self.get_new_posi(),1,1,1,1,'food')
        self.objects.append(food4)
        #
        pois = GridItem(self.get_new_posi(),1,1,0,-1,'fire')
        self.objects.append(pois)
        pois2 = GridItem(self.get_new_posi(),1,1,0,-1,'fire')
        self.objects.append(pois2)        
        #    
        return self.render()
        #
    
    def render(self):
        """
        """
        a = np.ones([self.sizeY+2, self.sizeX+2, 3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,\
              item.x+1:item.x+item.size+1,\
              item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]                    
        #
        self.state = a
        return a
        #
    
    def step(self, action):
        """
        """
        penalty = self.move_hero(action)
        reward, done = self.check_reward()
        state = self.render()
        #
        return state, (reward+penalty), done, None
    
    def close(self):
        pass
    #
    
    #
    def map_to_pic(self, state):
        """
        """
        b = scipy.misc.imresize(state[:,:,0],[84,84,1],interp='nearest')
        c = scipy.misc.imresize(state[:,:,1],[84,84,1],interp='nearest')
        d = scipy.misc.imresize(state[:,:,2],[84,84,1],interp='nearest')
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
        filename = "gridworld_temp.eps"  # eps < png < jpg
        plt.savefig(filename)
        pic_with_score = imageio.imread(filename) 
        os.remove(filename)
        #
        return pic, pic_with_score
        #
    
    def get_new_posi(self):
        """
        """
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        current_positions = []
        for object_a in self.objects:
            if (object_a.x,object_a.y) not in current_positions:
                current_positions.append((object_a.x,object_a.y))
        for pos in current_positions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]
        
    #
    def move_hero(self, direction):
        """
        """
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.0
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize
    
    def check_reward(self):
        """
        """
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        #
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GridItem(self.get_new_posi(),1,1,1,1,'food'))
                else: 
                    self.objects.append(GridItem(self.get_new_posi(),1,1,0,-1,'fire'))
                return other.reward, False
        else:
            return 0.0, False
        #

    

#
