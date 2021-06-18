import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from collections import namedtuple
import torch.optim as optim
import gym
# import preprocess
# import StockSimulation


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_dim, self.hidden_size)
        self.mean1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mean2 = nn.Linear(self.hidden_size, self.action_dim)
        self.v2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v3 = nn.Linear(self.hidden_size, 1)

    def forward(self, states):
        y1 = F.relu(self.fc1(states))

        mean = torch.tanh(self.mean1(y1))
        action_means = self.mean2(mean)

        value = F.relu(self.v2(y1))
        value = self.v3(value)

        return action_means, value


class Policy():

    def __init__(self, update_freq, lr=0.01, gamma=0.99, epsilon=0.2, k_epochs=100, std=1):

        self.update_freq = update_freq
        self.std = float(std)
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.gamma = gamma
        self.lr = lr

        self.old_policy = Net(
            env.observation_space.shape[0], env.action_space.shape[0])
        self.policy = Net(
            env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.mse = nn.MSELoss()

        self.rewards = []
        self.dones = []
        self.states = []
        self.actions = []
        self.log_probs = []

    def select_action(self, states):
        with torch.no_grad():

            states = torch.FloatTensor(states)
            action_means,  _ = self.old_policy.forward(states)
            variance = torch.tensor([self.std*self.std])
            cov_matrix = torch.diag(variance)

            m = MultivariateNormal(action_means, cov_matrix)
            action = m.sample()
            action = torch.clamp(action, -2, 2)

            self.states.append(states)
            self.actions.append(action)
            self.log_probs.append(m.log_prob(action))

        return action.item()

    def evaluate(self, states, actions):

        # states = torch.tensor(states,  dtype=torch.float32)
        # print(states.size())
        action_means, state_values = self.policy.forward(states)
        # print(action_means.size())
        actions = torch.reshape(actions, (self.update_freq, 1))
        actions = torch.clamp(actions, -2, 2)
        variance = torch.tensor([self.std*self.std],  dtype=torch.float32)
        variance = variance.expand_as(action_means)
        print(action_means)
        cov_matrix = torch.diag_embed(variance)
        print("cov_matrix: \n{}".format(cov_matrix))
        m = MultivariateNormal(action_means, cov_matrix)
        action_logprobs = m.log_prob(actions)

        return action_logprobs, state_values

    def clear(self):
        # del self.saved_actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]

    def update(self):

        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done == 1:
                discounted_reward = 0
            else:
                discounted_reward = reward + self.gamma*discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards,  dtype=torch.float32).detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states = torch.stack(self.states).squeeze()
        actions = torch.stack(self.actions).squeeze()
        old_probs = torch.stack(self.log_probs).squeeze()

        for _ in range(self.k_epochs):

            log_probs, values = self.evaluate(states, actions)
            ratios = torch.exp(log_probs-old_probs.detach())
            values = torch.reshape(values, (1, self.update_freq)).squeeze()
            advantages = rewards - values.detach()
            first = ratios*advantages
            second = torch.clamp(ratios, 1-self.epsilon,
                                 1+self.epsilon)*advantages
            surrogate = torch.min(first, second)
            loss = (-0.5)*surrogate + 0.5*self.mse(values, rewards)

            optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        if self.lr > 0.00001:
            self.lr *= 0.99
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.clear()


def learn():
    update_freq = 10
    model = Policy(update_freq)
    count = 0
    for epi in range(2000):
        state = env.reset()
        ep_reward = 0
        ep_count = 0
        while True:
            # env.render()
            count += 1
            ep_count += 1
            action = model.select_action(state)
            next_state,  reward, done, _ = env.step([action])
            ep_reward += reward
            model.rewards.append(reward)
            model.dones.append(done)
            if count % 1000 == 0:
                temp = model.std-0.05
                model.std = max(temp, 0.1)
            if count % update_freq == 0:
                model.update()
            if done == True or ep_count > 200:
                break
            state = next_state
        print("epi: {}, count: {} , reward: {}".format(epi, count, ep_reward))


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    learn()
