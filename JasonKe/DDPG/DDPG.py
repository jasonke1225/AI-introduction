import matplotlib.pyplot as plt
import gym
import torch
import math
import numpy as np
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import DDPG.replaymemory as replaymemory
from DDPG.OUNoise import OUNoise

import time

'''
紀錄q直 try
'''

# replaymemory state, next_state修改 in discrete


criterion = nn.MSELoss()

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class policyNetwork(nn.Module):
    def __init__(self, num_state, num_action, init_w = 3e-3):
        nn.Module.__init__(self)

        self.num_state = num_state
        self.num_action = num_action

        hidden_size = 256
        self.input_layer = nn.Linear(num_state, hidden_size)

        self.linear_layer = nn.Linear(hidden_size, hidden_size)

        self.layer_output = nn.Linear(hidden_size, num_action)


        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.input_layer.weight.data = fanin_init(self.input_layer.weight.data.size())
        self.linear_layer.weight.data = fanin_init(self.linear_layer.weight.data.size())
        self.layer_output.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        #print(state)
        if torch.cuda.is_available():
            state = state.cuda()
        state = self.input_layer(state)
        state = F.relu(state)
        state = self.linear_layer(state)
        state = F.relu(state)
        state = self.layer_output(state)
        
        ### 為甚麼

        state = F.tanh(state)

        return state.view(-1, self.num_action)

class valueNetwork(nn.Module):
    def __init__(self, num_state, num_action, init_w = 3e-4):
        nn.Module.__init__(self)

        self.num_state = num_state
        self.num_action = num_action
        
        hidden_size = 256

        self.input_layer = nn.Linear(num_state, hidden_size)
        self.linear_layer = nn.Linear(hidden_size + num_action, hidden_size)
        self.layer_output = nn.Linear(hidden_size, 1)

        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.input_layer.weight.data = fanin_init(self.input_layer.weight.data.size())
        self.linear_layer.weight.data = fanin_init(self.linear_layer.weight.data.size())
        self.layer_output.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, ac):
        if torch.cuda.is_available():
            state = state.cuda()
        state = self.input_layer(state)
        state = F.relu(state)
        state = self.linear_layer(torch.cat([state, ac], 1))
        state = F.relu(state)
        state = self.layer_output(state)

        return state.view(-1, 1)



class DDPG():
    def __init__(self, env, learning_rate):
        self.num_action = env.action_space.shape[0]
        self.observation_space = env.observation_space.shape[0]

        self.pZ = policyNetwork(num_state=self.observation_space, num_action=self.num_action)
        self.pZ_target = policyNetwork(num_state=self.observation_space, num_action=self.num_action)
        self.policyoptimizer = optim.Adam(self.pZ.parameters(), 1e-4)

        self.vZ = valueNetwork(num_state=self.observation_space, num_action=self.num_action)
        self.vZ_target = valueNetwork(num_state=self.observation_space, num_action=self.num_action)
        self.valueoptimizer = optim.Adam(self.vZ.parameters(), 1e-3)

        if torch.cuda.is_available():
            self.pZ.cuda()
            self.pZ_target.cuda()
            self.vZ.cuda()
            self.vZ_target.cuda()


        self.tau = 0.001
        self.batch_size = 256
        self.eps = 1.0
        self.discount = 0.99
        self.replaymemory = replaymemory.replaymemory(self.observation_space,self.num_action,1000000)
        self.n_iter = 5000
        self.noise = OUNoise(self.num_action)
        # self.memory_enough = False

        self.action_low, self.action_high = env.action_space.low, env.action_space.high
        

    def get_action(self, state):
        with torch.no_grad():
            action = self.pZ.forward(state)[0].cpu() + self.eps*torch.Tensor([self.noise.noise()])
            action.clamp(min = self.action_low[0], max= self.action_high[0])
            return action

    def soft_update(self):
        for target_param, param in zip(self.pZ_target.parameters(),self.pZ.parameters()):
            target_param.data.copy_(self.tau*param + (1.0-self.tau)*target_param)

        for target_param, param in zip(self.vZ_target.parameters(),self.vZ.parameters()):
            target_param.data.copy_(self.tau*param + (1.0-self.tau)*target_param)

    def load_model(self,name,episode):
        self.pZ.load_state_dict(torch.load('./preTrained/{}/DDPG_Actor_{}.pth'.format(name,episode)))
        self.pZ_target.load_state_dict(torch.load('./preTrained/{}/DDPG_TargetActor_{}.pth'.format(name,episode)))
        self.vZ.load_state_dict(torch.load('./preTrained/{}/DDPG_Critic_{}.pth'.format(name,episode)))
        self.vZ_target.load_state_dict(torch.load('./preTrained/{}/DDPG_TargetCritic_{}.pth'.format(name,episode)))

    def save_model(self,name,episode):
        torch.save(self.pZ.state_dict(), './preTrained/{}/DDPG_Actor_{}.pth'.format(name,episode))
        torch.save(self.pZ_target.state_dict(), './preTrained/{}/DDPG_TargetActor_{}.pth'.format(name,episode))
        torch.save(self.vZ.state_dict(), './preTrained/{}/DDPG_Critic_{}.pth'.format(name,episode))
        torch.save(self.vZ_target.state_dict(), './preTrained/{}/DDPG_TargetCritic_{}.pth'.format(name,episode))

    def calculate_loss(self, samples):
        theta = self.vZ(samples['states'],samples['actions'])

        with torch.no_grad():
            Z_next = self.vZ_target(samples['next_states'],self.pZ_target(samples['next_states']))
            dones = samples['dones']
            target_theta = samples['rewards'] + self.discount *(1-dones)* Z_next
            
        loss = criterion(theta, target_theta)
        
        return loss

    def learn(self, env, th):
        state = env.reset()
        self.noise.reset()

        ### 使動作noise收斂
        # self.eps = self.eps - 0.001 if self.eps > 0.001 else self.eps

        score = 0

        
        for i in range(self.n_iter):
            #self.eps_annealing()
            action = self.get_action(torch.Tensor([state]))
            #print(action)
            env.render()

            next_state, reward, done, _ = env.step(action)

            score += reward

            self.replaymemory.push(
                state, action, next_state, reward, done)

            state = next_state
            if done:
                state = env.reset()

            while len(self.replaymemory) < self.batch_size:
                action = self.get_action(torch.Tensor([state]))
                next_state, reward, done, _ = env.step(action)
                self.replaymemory.push(state, action, next_state, reward, done)
                state = next_state

            samples = self.replaymemory.get_samples(self.batch_size)
            #print(samples)
            

            ### critic update
            self.valueoptimizer.zero_grad()
            loss = self.calculate_loss(samples)
            loss.backward()
            self.valueoptimizer.step()

            ### actor update
            policyloss = -self.vZ(samples['states'],self.pZ(samples['states'])).mean()
            #print(policyloss)
            self.policyoptimizer.zero_grad()
            policyloss.backward()
            self.policyoptimizer.step()

            
            #if self.memory_enough:
                #L.write(str(float(loss))+"\n")

            self.soft_update()
        

            if done:
                # with torch.no_grad():
                #     s = env.reset()
                #     a = self.pZ(torch.Tensor([s]))
                #     print(i,'   ',a)
                #     print('      ',self.vZ(torch.Tensor([s]), a))
                break

        return float(score)
        
