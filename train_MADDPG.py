import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pettingzoo.mpe import simple_tag_v2
import supersuit as ss
from pettingzoo.mpe import simple_tag_v3


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        # Give the desired size for the output layer
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(torch.tensor(state)))
        x = torch.relu(self.fc2(x))
        # Obtain the action probabilities
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        # Fill in the desired dimensions
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(torch.tensor(state)))
        # Calculate the output value
        value = self.fc2(x)
        return value


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self, obs, actions, rewards, next_obs, done):
        data = (obs, actions, rewards, next_obs, done)
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def train(environment, max_episode_count, max_episode_length):
    env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=1000)
    observations, infos = env.reset()
    # create a tuple of agent networks
    actors = [Actor() for i in range(env.num_agents-1)]
    critics = [Critic() for i in range(env.num_agents-1)]
    for episode in range(max_episode_count):
        # initialize random process for action exploration
        observations, info = env.reset()
        # receive initial state x
        for t in range(max_episode_length):
            for agent in agents:
                # select action a_i w.r.t. current action policy + exploration
            # execute actions
            # observe reward and new state
            # store in replay buffer
            # update new state
            for agent in agents:
                # sample random minibatch (S samples) from D (a.k.a. Replay Buffer)
                # set y (target for each agent)
                # update critic by minimizing loss
                # update actor using sampled policy gradient
            # update target network parameters for all agents