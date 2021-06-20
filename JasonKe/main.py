#%%
import gym
import time
import os
import numpy as np
from DDPG.DDPG import DDPG
from DQN.DQN import DQN
from preprocess import arrangeData, getArrangedData
from StockSimulation import StockEnv as SE

env_name = 'StockSimulation'

def train_model(df):

    '''
    TODO :  create your agent and do training here.
            please save your parameter
    '''
    env = SE(df, 20150808, 20190514)

    # agent = DDPG(env=env, learning_rate=1e-3)
    agent = DDPG(env=env)
    pre_n_train, n_train = 0, 50
    # agent.load_model(env_name, pre_n_train)
    ewma_reward = None
    for i in range(pre_n_train, n_train):
        score = agent.learn(env,i)
        ewma_reward = ewma_reward * 0.94 + score * 0.06 if ewma_reward is not None else score
        print('the '+str(i)+' episode, reward: '+str(score)+', ewma reward: '+ str(ewma_reward))

        if i % 1 == 0:
            agent.save_model(env_name, i)

def validation(df):

    env = SE(df, 20190515, 20200514)
    
    agent = DDPG(env=env)
    
    for epi in range(0, 105, 10):
        agent.load_model(env_name, epi)
        score = agent.test(env)
        print('the '+str(epi)+' episode, reward: '+str(score))

def test_model(df):
    env = SE(df, 20190515, 20210514)
    
    agent = DDPG(env=env)
    for epi in range(0, 10, 1):
        agent.load_model(env_name, epi)
        score = agent.test(env)
        print('the '+str(epi)+' episode, reward: '+str(score))

def main():
    # arrangeData('primevalData')
    df = getArrangedData()

    train_model(df)
    # validation(df)
    # test_model(df)

if __name__ == '__main__':
    main()