import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DQN.replaymemory import replaymemory

# replaymemory state, next_state修改 in discrete
loss_func = nn.MSELoss()


class Network(nn.Module):
    def __init__(self, num_state, num_action):
        nn.Module.__init__(self)

        self.num_state = num_state
        self.num_action = num_action

        self.input_layer = nn.Linear(num_state, 512)
        self.layer_output = nn.Linear(512, num_action)

    def forward(self, state):
        #print(state)
        if torch.cuda.is_available():
            state = state.cuda()
        state = self.input_layer(state)
        state = F.relu(state)
        state = self.layer_output(state)

        return state.view(-1, self.num_action)


class DQN():
    def __init__(self, env, learning_rate):
        self.num_action = 729
        self.observation_space = env.observation_space.shape[0]
        self.Z = Network(num_state = self.observation_space, num_action = self.num_action)
        self.Z_target = Network(num_state = self.observation_space, num_action = self.num_action)
        self.optimizer = optim.Adam(self.Z.parameters(), learning_rate)

        if torch.cuda.is_available():
            self.Z.cuda()
            self.Z_target.cuda()


        self.batch_size = 32
        self.step = 0
        self.eps = 0.05
        self.discount = 0.95
        self.replaymemory = replaymemory(self.observation_space,self.num_action,10000)
        self.n_iter = 5000
        self.memory_enough = False

    def get_action(self, state, eps_choose = True):
        with torch.no_grad():
            action = 0
            if np.random.uniform() < self.eps and eps_choose == True:
                action = np.random.randint(0, 729)
            else:
                action = int(self.Z.forward(state).max(1)[1])
            
            return action
            
    def calculate_loss(self, samples):
        theta = self.Z(samples['states'])[np.arange(self.batch_size), samples['actions']]
        with torch.no_grad():
            # target_theta為我們想逼近的最終理想distribution
            Z_nexts = self.Z_target(samples['next_states'])
            Z_next = Z_nexts[np.arange(self.batch_size), Z_nexts.argmax(1)]
            dones = samples['dones'].view(-1, self.batch_size).squeeze()
            target_theta = samples['rewards'].view(-1, self.batch_size).squeeze() + self.discount *(1-dones)* Z_next
            
        loss = loss_func(theta, target_theta)

        return loss
    
    def learn(self, env, th):
        state = env.reset()
        score = 0
        
        for i in range(self.n_iter):
            # if i % 500 == 0:
            #     print(i)
            #self.eps_annealing()
            action = self.get_action(torch.Tensor([state]))
            action_continuous = env.convert_action(action)
            # print(action)
            next_state, reward, done, _ = env.step(action_continuous)
            score += reward
            #env.render()

            self.replaymemory.push(
                state, action, next_state, reward, done)
            if done:
                state = env.reset()
            else:
                state = next_state

            if len(self.replaymemory) < 1000:
                continue
            self.memory_enough = True

            samples = self.replaymemory.get_samples(self.batch_size)
            #print(samples)
            loss = self.calculate_loss(samples)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            
            if self.step % 100 == 0:
                self.Z_target.load_state_dict(self.Z.state_dict())
                self.Z_target.load_state_dict(self.Z.state_dict())
                
            self.step += 1
        
            if done or i == self.n_iter-1:
                with torch.no_grad():
                    print(i,self.Z_target(torch.Tensor([env.reset()])).max(1))
                break
        
        return float(score)