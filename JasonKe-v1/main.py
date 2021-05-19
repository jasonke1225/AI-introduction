#%%

import gym
import time
import numpy as np
from DDPG.DDPG import DDPG
from preprocess import arrangeData, getArrangedData
from StockSimulation import StockEnv as SE

# arrangeData('primevalData')
df = getArrangedData()

env_name = 'StockSimulation'
env = SE(df, 20150808)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(obs_dim,act_dim)

### `th` times multitraining for stats
th = 1
for tt in range(0,th):
    '''
    TODO :  create your agent and do training here.
            please save your parameter
    '''
    # agent = DDPG(env=env, learning_rate=1e-3)
    # pre_n_train, n_train = 0, 1000
    # # agent.load_model(env_name, pre_n_train)
    # ewma_reward = 0
    # for i in range(pre_n_train, n_train):
    #     score = agent.learn(env,i)
    #     ewma_reward = ewma_reward * 0.96 + score * 0.04
    #     print('the '+str(i)+' episode, reward: '+str(score)+', ewma reward: '+ str(ewma_reward))

    # agent.save_model(env_name, n_train)
    
