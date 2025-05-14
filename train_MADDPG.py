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
    def __init__(self, obs_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        # Give the desired size for the output layer
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, obs):
        x = torch.relu(self.fc1(torch.tensor(obs)))
        x = torch.relu(self.fc2(x))
        # Obtain the action probabilities
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + num_actions, 64)
        # Fill in the desired dimensions
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state, actions):
        s = torch.tensor(state)
        a = torch.tensor(actions)
        x = torch.relu(self.fc1(torch.cat((s, a), dim=-1)))
        # Calculate the output value
        value = self.fc2(x)
        return value


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self, state, actions, rewards, next_state, obs):
        data = (state, actions, rewards, next_state, obs)
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

'''
def train(environment, max_episode_count, max_episode_length, max_capacity):
    env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=max_episode_length)
    observations, infos = env.reset()
    # create a tuple of agent networks
    # We need to find the actual state size and action size (to replace 16 & 5)
    actors = [Actor(16, 5) for i in range(env.num_agents-1)]
    critics = [Critic(16) for i in range(env.num_agents-1)]
    D = ReplayBuffer(max_capacity)
    while env.agents:
        # initialize random process for action exploration
        # receive initial state x
        observations, info = env.reset()
        for t in range(max_episode_length):
            for actor in actors:
                actor.forward()
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
'''