# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 09:11:50 2021

@author: Ethan
"""
print('=========================')

import numpy as np # used for arrays
import gym # pull the environment
import math # needed for calculations
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 100000
episode = 0
total_reward = 0
prior_reward = 0
best_reward = 0
simulation_done = False

update_period = 1000
render_period = 2500
policysave_period = 10000
save_q_table = False

Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
returns = []

epsilon = 1

epsilon_decay_value = 0.99995

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
#q_table = np.load('Q-Table.npy')
q_table.shape

def get_discrete_state(state):
    discrete_state = state/np_array_win_size + np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))

#for episode in range(EPISODES + 1): #go through the episodes
while simulation_done == False:
    discrete_state = get_discrete_state(env.reset()) #get the discrete start for the restarted environment 
    done = False
    episode_reward = 0 #reward starts as 0 for each episode

    if episode % update_period == 0: 
        print("Episode: " + str(episode))

    while not done: 
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) #take cordinated action
        else:
            action = np.random.randint(0, env.action_space.n) #do a random ation
            
        new_state, reward, done, _ = env.step(action) #step action to get new states, reward, and the "done" status.
        episode_reward += reward #add the reward
        new_discrete_state = get_discrete_state(new_state)

        if episode % render_period == 0: #render
            env.render()

        if not done: #update q-table
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state


    if epsilon > 0.05: #epsilon modification
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)

            if episode % 2500 == 0:
                print("Epsilon: " + str(epsilon))

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    if episode % update_period == 0: #every few episodes print the average reward
        mean_reward = total_reward / update_period
        returns.append(mean_reward)
        if (mean_reward > 200):
            mean_reward = 200
            iterations = range(0, episode + 1, update_period)
            plt.plot(iterations, returns)
            plt.ylabel('Average Return')
            plt.xlabel('Iterations')
            plt.ylim(top=250)
            simulation_done = True
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

    if (mean_reward > best_reward) and (episode > policysave_period):
        best_reward = mean_reward
        print("_____New Best Reward_____")
        if (save_q_table):
            np.save('Q-Table.npy', q_table)
            print("______Q-TABLE SAVED______")
    
    if episode % update_period == 0: 
        print('=========================')
    episode += 1

env.close()
