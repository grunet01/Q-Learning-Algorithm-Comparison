# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:58:17 2021

@author: Ethan
"""

import numpy as np # used for arrays
import gym # pull the environment

env = gym.make("CartPole-v1")
episodes = 10
max_episode_time = 1000000

total = 0

np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

q_table = np.load('Q-Table.npy')

def get_discrete_state(state):
    current_discrete_state = state/np_array_win_size + np.array([15,10,1,10])
    return tuple(current_discrete_state.astype(np.int))

def getQAction(state):
    discrete_state = get_discrete_state(state)
    action = np.argmax(q_table[discrete_state])
    return action

for i_episodes in range(episodes):
    observation = env.reset()
    for t in range(max_episode_time):
        env.render()
        action = getQAction(observation)
        observation, _, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            total+= t+1
            break
env.close()
average = total / episodes
print("Average timesteps: " + str(average))

