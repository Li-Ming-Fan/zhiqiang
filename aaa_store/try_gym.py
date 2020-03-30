
import time

import gym
env = gym.make('SpaceInvaders-v0')
# env = gym.make("CartPole-v1")
# env = gym.make("Gopher-v4")

env.reset()
env.render()

time.sleep(60)


observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()

