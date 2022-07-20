import gym
import numpy as np
import matplotlib

env = gym.make('FrozenLake8x8-v0')

#observation = env.reset()
duration = 50000 #number of episodes
maxTimesteps = 99 #maximum timesteps for each episode
printPeriod = 500

Q = np.zeros([env.observation_space.n,env.action_space.n])
    
# q learning params
eta = .628
gma = .9
def updateQ(s, s1, a, r):
    Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])


print('================================')
for episode in range(duration+1):
    state = env.reset()
    done = False
    episodeReturn = 0
    totalReturn = 0
    t = 0
    while t < maxTimesteps:
        #env.render()
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(t+1)))
        newState, reward, done, _ = env.step(action)
        updateQ(state, newState, action, reward)
        episodeReturn += reward
        state = newState
        t += 1
        if t == maxTimesteps - 1 or done:
            totalReturn += episodeReturn
            break
    if episode % printPeriod == 0:
        print('Number of Episodes: ', episode)
        print('Avg Episode return:', totalReturn / printPeriod)
        print('================================')
