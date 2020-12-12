#########################################
# Author: hexatorai
#  Date : 10/09/2019
#########################################
import torch
from torch import nn
import numpy as np
from collections import deque
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FC1_UNITS = 400  # fc 1 units
FC2_UNITS = 400  # fc layer 2 units
TARGET_UPDATE = 50  # number of steps to update the target

class DQN(nn.Module):

    def __init__(self, lr, state_size, fc1_units, fc2_units,
                 n_actions, weight_file_path="weights/weights"):
        super(DQN, self).__init__()

        self.state_size = state_size
        self.fc1 = nn.Linear(state_size, fc1_units)  # layer 1
        self.fc2 = nn.Linear(fc1_units, fc2_units)  # layer 2
        self.fc3 = nn.Linear(fc2_units, n_actions)  # layer 3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)

        self.weight_file_path = weight_file_path

        try:
            self.load_state_dict(torch.load(weight_file_path))
            print("weights are loaded")
        except:
            print("could not load the weight file")

    def forward(self, state):
        state = state.float()
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

    def save_weights(self, file_path):
        self.weight_file_path = file_path
        torch.save(self.state_dict(), file_path)


class Memory:

    def __init__(self, max_size):
        self.index = 0  # next index to store
        self.max_size = max_size  # max memory size
        self.memory = deque(maxlen=max_size)  # memory deque

    def push(self, state, next_state, action, reward, done):
        # extend until max memory
        if len(self.memory) < self.max_size:
            self.memory.append(None)

        self.memory[self.index] = (state, next_state, action, reward, done)
        self.index = (self.index + 1) % self.max_size  # update the index

    def sample(self, sample_size):
        batch = random.sample(self.memory, sample_size)
        return self.lbs(0, batch), self.lbs(1, batch), self.lbs(2, batch), self.lbs(3, batch), self.lbs(4, batch)

    def __len__(self):
        return len(self.memory)

    def lbs(self, pos, batch):
        return [b[pos] for b in batch]


class DQAgent:

    # =====================================================================
    # constructor
    #   lr         = learning rate
    #   gamma      = discount factor
    #   eps        = initial epsilon value
    #   eps_final  = the final value of the epsilon (minimum)
    #   mem_size   = capacity of the memory
    #   state_size = number of state variables 
    #   batch_size = size of the batch of experience to fit the model
    #   n_actions  =  no of actions
    #   target_update = number of steps to update the target net
    def __init__(self, lr, gamma, eps, eps_final, eps_dec,
                 mem_size, state_size, batch_size, n_actions):

        self.gamma = gamma
        self.eps = eps
        self.eps_final = eps_final
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.state_size = state_size
        self.mem_size = mem_size
        self.lr = lr
        self.n_actions = n_actions
        self.target_update = TARGET_UPDATE

        # the policy net 
        self.net = DQN(lr, state_size, FC1_UNITS, FC2_UNITS, n_actions)

        # the target net
        self.target_net = DQN(lr, state_size, FC1_UNITS, FC2_UNITS, n_actions)
        self.target_net.load_state_dict(self.net.state_dict())

        # the memory
        self.memory = Memory(mem_size)

        # number of learning steps
        self.step_count = 0

        # store losses of each step
        self.losses = []

    # Get the next Action
    def next_action(self, state, learn):
        if learn:
            if random.random() > self.eps:
                state_t = torch.tensor([state]).to(device)
                actions = self.net.forward(state_t)
                return torch.argmax(actions).item()

            return random.randint(0, self.n_actions - 1)
        else:
            if random.random() > self.eps:  # predict next action with the policy net
                state_t = torch.tensor([state]).to(device)
                actions = self.net.forward(state_t)
                return torch.argmax(actions).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.net.optimizer.zero_grad()

        # get the experience batch
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = \
            self.memory.sample(self.batch_size)

        # convert to torch tensors
        state_batch = torch.tensor(state_batch).to(device)
        next_state_batch = torch.tensor(next_state_batch).to(device)
        reward_batch = torch.tensor(reward_batch).to(device)
        done_batch = torch.tensor(done_batch).to(device)

        batch_index = np.arange(self.batch_size)

        # Q values for each state and action
        q_eval = self.net.forward(state_batch)[batch_index, action_batch]

        # Q values for each next state
        q_next = self.target_net.forward(next_state_batch)

        # for done states q_targer = reward. so make q_next value 0
        q_next[done_batch] = 0.0

        # Belleman equation
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        # fit the policy net
        loss = self.net.loss(q_target, q_eval).to(device)
        loss.backward()
        self.net.optimizer.step()

        # save the loss
        self.losses.append(loss.item())

        # update the target net
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.step_count += 1

        # reduce epsilon
        if self.eps > self.eps_final:
            self.eps -= self.eps_dec

        else:
            self.eps = self.eps_final
