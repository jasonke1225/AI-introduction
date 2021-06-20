import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from collections import namedtuple
import torch.optim as optim
import gym
from preprocess import arrangeData, getArrangedData
from StockSimulation import StockEnv as SE
import normalization
import datetime


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_dim, self.hidden_size)
        self.co = nn.Linear(self.hidden_size, self.hidden_size)
        self.mean = nn.Linear(self.hidden_size, action_dim)
        self.log_std = nn.Linear(self.hidden_size, action_dim)

    def forward(self, states):

        y1 = F.relu(self.fc1(states))
        co = F.relu(self.co(y1))
        action_mean = torch.tanh(self.mean(co))
        log_std = F.relu(self.log_std(co))
        std = torch.exp(log_std)
        return action_mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
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
        state_value = torch.tanh(self.v3(value))
        return state_value


class Policy():

    def __init__(self, update_freq, lr=1e-3, gamma=0.99, epsilon=0.2, k_epochs=100):

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
            stds = torch.clamp(stds, 0.001, 0.01)
            variance = stds*stds
            cov_matrix = torch.diag(variance)
            action_means = torch.clamp(action_means, -1, 1)
            m = MultivariateNormal(action_means, cov_matrix)
            action = m.sample()
            log_prob = m.log_prob(action)

            self.states.append(states)
            self.actions.append(action)
            self.log_probs.append(log_prob)

        return action.numpy()

    def evaluate(self, states, actions):

        action_means, stds = self.policy.forward(states)
        state_values = self.critic.forward(states)
        stds = torch.clamp(stds, 0.001, 0.01)
        variance = stds*stds
        cov_matrix = torch.diag_embed(variance)
        action_means = torch.clamp(action_means, -1, 1)
        m = MultivariateNormal(action_means,  cov_matrix)
        action_logprobs = m.log_prob(actions)
        distribution_entropy = m.entropy()

        return action_logprobs, state_values, distribution_entropy

    def clear(self):
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

        rewards = torch.tensor(
            rewards,  dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states = torch.stack(self.states).squeeze()
        actions = torch.stack(self.actions).squeeze()
        old_probs = torch.stack(self.log_probs).squeeze()

        for _ in range(self.k_epochs):

            log_probs, state_values, distribution_entropy = self.evaluate(
                states, actions)

            ratios = torch.exp(log_probs-old_probs.detach())
            advantages = rewards - state_values.detach()
            first = ratios*advantages
            second = torch.clamp(ratios, 1-self.epsilon,
                                 1+self.epsilon)*advantages
            surrogate = torch.min(first, second)

            actor_loss = -surrogate.mean() - 0.01*distribution_entropy.mean()

            state_values = torch.reshape(
                state_values, (self.update_freq, 1)).squeeze()
            critic_loss = 0.5*self.mse(state_values, rewards)
            loss = actor_loss+critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.lr > 0.00001:
            self.lr *= 0.99
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.clear()


def learn():
    update_freq = 10000
    model = Policy(update_freq)
    count = 0
    for epi in range(100):
        state = env.reset()
        ep_reward = 0
        ep_count = 0
        while True:
            count += 1
            ep_count += 1
            temp_state = normalization.normalize(state)
            action = model.select_action(temp_state)
            next_state,  reward, done, _ = env.step(action)
            ep_reward += reward
            model.rewards.append(reward)
            model.dones.append(done)
            if count % update_freq == 0:
                model.update()
            if done == True:
                break
            state = next_state
        if epi % 10 == 0:
            t = datetime.datetime.now()
            t = str(t)[11:19]
            t = t.replace(":", "_")
            torch.save(model.old_policy.state_dict(),
                       './preTrained/PPO_actor_{}.pth'.format(epi))
            torch.save(model.critic.state_dict(),
                       './preTrained/PPO_critic_{}.pth'.format(epi))

        print("epi: {}, count: {} , reward: {}".format(epi, count, ep_reward))


def test():
    for epi in range(0, 91, 10):
        model = Policy(1)
        model.old_policy.load_state_dict(torch.load(
            './preTrained/PPO_actor_{}.pth'.format(epi)))
        model.policy.load_state_dict(torch.load(
            './preTrained/PPO_actor_{}.pth'.format(epi)))
        model.critic.load_state_dict(torch.load(
            './preTrained/PPO_critic_{}.pth'.format(epi)))
        state = env.reset()
        ep_reward = 0
        ep_count = 0
        while True:
            ep_count += 1
            temp_state = normalization.normalize(state)
            action = model.select_action(temp_state)
            next_state,  reward, done, _ = env.step(action)
            ep_reward += reward
            model.rewards.append(reward)
            model.dones.append(done)
            if done == True:
                break
            state = next_state
        print("epi: {} , reward: {}".format(epi, ep_reward))

    return


if __name__ == "__main__":

    df = getArrangedData()
    env_name = 'StockSimulation'
    # env = SE(df, 20150808, 20190514)
    env = SE(df, 20190514, 20210514)
    # learn()
    test()
