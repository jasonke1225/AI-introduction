# Spring 2021, IOC 5269 Reinforcement Learning
# HW2: REINFORCE with baseline and A2C

import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size1 = 400
        self.hidden_size2 = 200
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size1)

        self.layer1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.layer2 = nn.Linear(self.hidden_size1, self.hidden_size2)

        self.action_layer = nn.Linear(self.hidden_size2, self.action_dim)
        self.value_layer = nn.Linear(self.hidden_size2, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        state = self.shared_layer(state)
        state = F.relu(state)
        state1 = self.layer1(state)
        state1 = F.relu(state1)
        action_prob = self.action_layer(state1)
        action_prob = self.softmax(action_prob)
        
        state2 = self.layer2(state)
        state2 = F.relu(state2)
        state_value = self.value_layer(state2)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        action_prob, state_value = self.forward(torch.Tensor([state]))
        m = Categorical(action_prob)
        action = m.sample()

        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        gamma_t = 1
        for idx in range(len(self.rewards)):
            V_next = saved_actions[idx+1].value if idx < len(self.rewards) -1 else 0
            td = self.rewards[idx] + V_next -saved_actions[idx].value

            p_loss = -1* gamma_t * saved_actions[idx].log_prob * td
            v_loss = td * td 
            policy_losses.append(p_loss)
            value_losses.append(v_loss)

            gamma_t *= gamma

        policy_loss = torch.cat(policy_losses).sum()
        value_loss = torch.cat(value_losses).sum()
        loss = policy_loss + value_loss

        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        
        # For each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        while t < 10000:
            t += 1

            action = model.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            # env.render()
            ep_reward += reward
            state = next_obs
            model.rewards.append(reward)

            if done:
                break
        
        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward(retain_graph=True)
        optimizer.step()

        model.clear_memory()

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()
        
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))
        # check if we have "solved" the cart pole problem
        if ewma_reward > env.spec.reward_threshold or i_episode>2000:
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20
    lr = 0.003
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    env.seed(random_seed)  
    print(env.spec.reward_threshold)
    torch.manual_seed(random_seed)  
    train(lr)
    test('LunarLander_0.01.pth')
    

