import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import torch.optim as optim
import gym

# SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# device = torch.device('cpu')

# if(torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.action_dim)
        self.v2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v3 = nn.Linear(self.hidden_size, 1)

    def forward(self, states):
        y1 = F.relu(self.fc1(states))
        y2 = F.relu(self.fc2(y1))
        y3 = self.fc3(y2)
        value2 = F.relu(self.v2(y1))
        value = self.v3(value2)
        action_probs = F.softmax(y3, dim=-1)
        return action_probs, value


class Policy():

    def __init__(self, lr=0.01, gamma=0.99, epsilon=0.2, k_epochs=80):

        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.gamma = gamma
        self.old_policy = Net(
            env.observation_space.shape[0], env.action_space.n)
        self.policy = Net(
            env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse = nn.MSELoss()

        # self.saved_actions = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.actions = []
        self.log_probs = []

    def select_action(self, states):

        with torch.no_grad():

            states = torch.FloatTensor(states)
            action_probs, state_value = self.old_policy.forward(states)
            m = Categorical(action_probs)
            action = m.sample()

            self.states.append(states)
            self.actions.append(action)
            self.log_probs.append(m.log_prob(action))

        return action.item()

    def evaluate(self, states, actions):

        states = torch.FloatTensor(states)
        action_probs, state_values = self.policy.forward(states)
        m = Categorical(action_probs)
        action_logprobs = m.log_prob(actions)

        return action_logprobs, state_values

    def clear(self):
        # del self.saved_actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]

    def update(self, buffer_size):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done == 1:
                discounted_reward = 0
            else:
                discounted_reward = reward + self.gamma*discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards,  dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states = torch.stack(self.states).squeeze()
        actions = torch.stack(self.actions).squeeze()
        old_probs = torch.stack(self.log_probs).squeeze()

        for _ in range(self.k_epochs):
            log_probs, values = self.evaluate(states, actions)
            ratios = torch.exp(log_probs-old_probs.detach())
            values = torch.reshape(values, (1, buffer_size)).squeeze()
            advantages = rewards - values.detach()
            first = ratios*advantages
            second = torch.clamp(ratios, 1-self.epsilon,
                                 1+self.epsilon)*advantages
            surrogate = torch.min(first, second)
            loss = (-0.5)*surrogate + 0.5*self.mse(values, rewards)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.clear()


def learn():
    model = Policy()
    count = 0
    update_freq = 1000
    for epi in range(2000):
        state = env.reset()
        ep_reward = 0
        ep_count = 0
        while True:
            # env.render()
            count += 1
            ep_count += 1
            action = model.select_action(state)
            # print(action)
            next_state,  reward, done, _ = env.step(action)
            ep_reward += reward
            model.rewards.append(reward)
            model.dones.append(done)
            if count % update_freq == 0:
                model.update(update_freq)
            if done == True:
                break
            state = next_state
        print("epi: {}, count: {} , reward: {}".format(epi, count, ep_reward))


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    learn()
# CartPole-v0
