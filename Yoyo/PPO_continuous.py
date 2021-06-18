import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple
import torch.optim as optim
import gym
# import preprocess
# import StockSimulation


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_dim, self.hidden_size)
        self.co = nn.Linear(self.hidden_size, self.hidden_size)
        self.mean = nn.Linear(self.hidden_size, 1)
        self.log_std = nn.Linear(self.hidden_size, 1)

    def forward(self, states):

        y1 = F.relu(self.fc1(states))
        co = F.relu(self.co(y1))
        action_mean = self.mean(co)
        log_std = F.relu(self.log_std(co))
        std = torch.exp(log_std)
        return action_mean.squeeze(), std.squeeze()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_dim, self.hidden_size)
        self.v2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v3 = nn.Linear(self.hidden_size, 1)

    def forward(self, states):
        y1 = F.relu(self.fc1(states))
        value = F.relu(self.v2(y1))
        state_value = self.v3(value)
        return state_value.squeeze()


class Policy():

    def __init__(self, update_freq, lr=0.001, gamma=0.99, epsilon=0.2, k_epochs=100):

        self.update_freq = update_freq
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.gamma = gamma
        self.lr = lr
        self.old_policy = Actor(
            env.observation_space.shape[0], env.action_space.shape[0])
        self.policy = Actor(
            env.observation_space.shape[0], env.action_space.shape[0])
        self.critic = Critic(
            env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ])
        self.mse = nn.MSELoss()

        self.rewards = []
        self.dones = []
        self.states = []
        self.actions = []
        self.log_probs = []

    def select_action(self, states):

        with torch.no_grad():

            states = torch.FloatTensor(states)
            action_means, stds = self.old_policy.forward(states)
            stds = torch.clamp(stds, 0.01, 1)
            m = Normal(action_means, stds)
            action = m.sample()
            log_prob = m.log_prob(action)

            self.states.append(states)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            # print("std: {}".format(stds))
            # print("action_mean: {}".format(action_means))
            # print("state: {}".format(states))
            # print("action: {}".format(action))
            # print("log_prob: {}".format(log_prob))
            # print()

        return action.item()

    def evaluate(self, states, actions):

        action_means, stds = self.policy.forward(states)
        state_values = self.critic.forward(states)
        stds = torch.clamp(stds, 0.01, 1)
        m = Normal(action_means,  stds)
        action_logprobs = m.log_prob(actions)
        # print(action_logprobs.size())
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

        # rewards = torch.tensor(rewards,  dtype=torch.float32)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states = torch.stack(self.states).squeeze().detach()
        actions = torch.stack(self.actions).squeeze().detach()
        old_probs = torch.stack(self.log_probs).squeeze().detach()

        for _ in range(self.k_epochs):
            log_probs, state_values = self.evaluate(states, actions)
            ratios = torch.exp(log_probs-old_probs.detach())
            # state_values = torch.reshape(
            #     state_values, (1, self.update_freq)).squeeze()
            advantages = rewards - state_values.detach()
            first = ratios*advantages
            second = torch.clamp(ratios, 1-self.epsilon,
                                 1+self.epsilon)*advantages
            surrogate = torch.min(first, second)
            actor_loss = -surrogate.mean()
            critic_loss = self.mse(state_values, rewards)
            loss = actor_loss+critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.lr > 0.000001:
            self.lr *= 0.999
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.clear()


def learn():
    update_freq = 10000
    model = Policy(update_freq)
    count = 0
    for epi in range(20000):
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
            if count % update_freq == 0:
                model.update()
            if done == True:
                break
            state = next_state

        print("epi: {}, count: {} , reward: {}".format(epi, count, ep_reward))


if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    learn()
