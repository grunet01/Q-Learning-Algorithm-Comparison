# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 09:11:50 2021

@author: Ethan
"""

import numpy as np # used for arrays
import gym # pull the environment
import math # needed for calculations
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

learning_rate = 0.2
discount_factor = 0.95
epsilon = 0.8
epsilon_decay_value = 0.99995
minimum_epsilon = 0.10
min_eps = 0
episodes = 25000

episode = 1
total_reward = 0
prior_reward = 0
best_reward = -250
simulation_done = False

update_period = 100
render_period = 1000
policysave_period = 100000
save_q_table = False

Observation = [18, 14]
np_array_win_size = np.array([10, 100])
returns = []

reduction = (epsilon - min_eps)/episodes

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape

def get_discrete_state(state):
    discretized_state = (state - env.observation_space.low) * np_array_win_size
    return tuple(np.round(discretized_state, 0).astype(int))

def get_custom_reward(state, new_state):
    custom_reward = 100*((math.sin(3*new_state[0,0]) * 
                   0.0025 + 0.5 * new_state[0,1] * 
                   new_state[0,1]) - 
                  (math.sin(3*state[0,0]) * 
                   0.0025 + 0.5 * state[0,1] * state[0,1]))
    return custom_reward * 2000

# for episode in range(episodes + 1): #go through the episodes
while not simulation_done:
    continuous_state = env.reset()
    discrete_state = get_discrete_state(continuous_state) #get the discrete start for the restarted environment 
    done = False
    episode_reward = 0 #reward starts as 0 for each episode

    if episode % update_period == 0: 
        print("Episode: " + str(episode))

    while not done: 
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) #take cordinated action
        else:
            action = np.random.randint(1, env.action_space.n) #do a random ation
            
        new_state, reward, done, _ = env.step(action) #step action to get new states, reward, and the "done" status.
        episode_reward += reward #add the reward
        new_discrete_state = get_discrete_state(new_state)

        if episode % render_period == 0: #render
            env.render()

        if not done: #update q-table
            energy_reward = get_custom_reward(np.reshape(continuous_state, (1, 2)),
                                              np.reshape(new_state, (1, 2)))
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - learning_rate) * current_q + learning_rate * (energy_reward + discount_factor * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state


    if epsilon > minimum_epsilon and epsilon > min_eps: #epsilon modification
        epsilon = epsilon - reduction
        if episode % 2500 == 0:
            print("Epsilon: " + str(epsilon))
    else:
        epsilon = minimum_epsilon

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    if (episode % update_period == 0): 
        mean_reward = total_reward / update_period
        total_reward = 0
        returns.append(mean_reward)
        print("Mean Environment Reward: " + str(mean_reward))
        
        if (mean_reward > -110):
            iterations = range(update_period, episode, update_period)
            plt.plot(iterations, returns)
            plt.ylabel('Average Return')
            plt.xlabel('Iterations')
            plt.ylim(top=0)
            simulation_done = False
        
        if (mean_reward > best_reward):
            best_reward = mean_reward
            print("_____New Best Reward_____")
    
    episode += 1

env.close()

