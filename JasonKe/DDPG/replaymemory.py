import torch
import random


class replaymemory():
    def __init__(self, observation_space, num_action, capacity):
        self.capacity = capacity
        self.data_set = []
        self.observation_space = observation_space
        self.num_action = num_action

    def push(self, state, action, next_state, reward, done):
        if torch.cuda.is_available():
            state_ = torch.Tensor([state]).cuda()
            action_ = action.cuda()
            next_state_ = torch.Tensor([next_state]).cuda()
            reward_ = torch.Tensor([reward]).cuda() 
            done_ = torch.Tensor([done]).cuda()
        transition = state_, action_, next_state_, reward_, done_
        self.data_set.append(transition)
        if len(self.data_set) > self.capacity:
            del self.data_set[0]

    def get_samples(self, batch_size):
        batch_datas = random.sample(self.data_set, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(
            *batch_datas)


        # state size = (-1, num_state)
        # 根據space做變更
        batch_state = torch.cat(batch_state).view(-1, self.observation_space)
        batch_action = torch.cat(batch_action).view(-1, self.num_action)
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)
        batch_next_state = torch.cat(batch_next_state).view(-1, self.observation_space)

        samples = {'states': batch_state, 'actions': batch_action, 'rewards': batch_reward.unsqueeze(
            1), 'next_states': batch_next_state, 'dones': batch_done.unsqueeze(1)}

        return samples

    def __len__(self):
        return len(self.data_set)
