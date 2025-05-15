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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        # Give the desired size for the output layer
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, obs: torch.Tensor):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        # Obtain the action probabilities
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_size, actions_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + actions_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        # Concatenate state and actions, no in-place modifications here
        x = torch.cat((state, actions), dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))  # Apply ReLU, not in-place
        value = self.fc2(x)
        return value


class MADDPGAgent:
    def __init__(self, obs_size, action_size, state_size, joint_action_size, lr=0.001):
        self.actor = Actor(obs_size, action_size).to(device)
        self.critic = Critic(state_size, joint_action_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def get_action(self, obs, model_out=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.actor(obs_tensor)
        dist = torch.distributions.Categorical(logits)
        action = dist.sample().item()
        if model_out:
            return action, logits.squeeze(0)
        return action

    def critic_value(self, state_list, action_list):
        return self.critic(state_list, action_list).squeeze(1)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, rewards, next_state, obs):
        data = (state, actions, rewards, next_state, obs)
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def one_hot(index, size):
    vec = torch.zeros(size)
    vec[index] = 1.0
    return vec


def train(max_episode_count, max_episode_length, max_capacity, minibatch_size, gamma, lr):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=max_episode_length)
    observations, infos = env.reset()
    adversaries = env.agents[0:-1]
    agents = {adv: MADDPGAgent(len(observations[adv]), env.action_space(adv).n, len(env.state()), sum(env.action_space(a).n for a in adversaries), lr) for adv in adversaries}
    D = ReplayBuffer(max_capacity)
    for i in range(max_episode_count):
        observations, infos = env.reset()
        while env.agents:
            state = env.state()
            actions = {}
            logits_store = {}
            for adv in adversaries:
                action, logits = agents[adv].get_action(observations[adv], model_out=True)
                actions[adv] = action
                logits_store[adv] = logits
            actions["agent_0"] = env.action_space("agent_0").sample()
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            next_state = env.state()
            D.push(state, actions, rewards, next_state, observations)
            observations = next_obs
            if len(D.buffer) >= minibatch_size:
                for adv in adversaries:
                    samples = D.sample(minibatch_size)
                    # Unpack and stack batches
                    s_states = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in samples]).to(device)
                    s_next_states = torch.stack([torch.tensor(s[3], dtype=torch.float32) for s in samples]).to(device)
                    s_rewards = torch.tensor([s[2][adv] for s in samples], dtype=torch.float32).to(device)
                    next_acts = []
                    for ag in adversaries:
                        obs_batch = torch.stack([torch.tensor(s[4][ag], dtype=torch.float32) for s in samples]).to(
                            device)
                        logits = agents[ag].actor(obs_batch)
                        next_action_indices = torch.multinomial(logits, 1).squeeze(1)
                        next_actions_onehot = F.one_hot(next_action_indices, num_classes=env.action_space(ag).n).float()
                        next_acts.append(next_actions_onehot)
                    next_joint_actions = torch.cat(next_acts, dim=-1)
                    with torch.no_grad():
                        target_q = s_rewards + gamma * agents[adv].critic(s_next_states, next_joint_actions).squeeze(1)
                    joint_acts = []
                    for ag in adversaries:
                        actions = torch.tensor([s[1][ag] for s in samples], dtype=torch.int64)
                        onehots = F.one_hot(actions, num_classes=env.action_space(ag).n).float().to(device)
                        joint_acts.append(onehots)
                    current_joint_actions = torch.cat(joint_acts, dim=-1)
                    current_q = agents[adv].critic(s_states, current_joint_actions).squeeze(1)
                    critic_loss = nn.MSELoss()(current_q, target_q)
                    agents[adv].update_critic(critic_loss)
                    # actor update
                    actor_acts = []
                    for ag in adversaries:
                        obs_batch = torch.stack([torch.tensor(s[4][ag], dtype=torch.float32) for s in samples]).to(device)
                        logits = agents[ag].actor(obs_batch)
                        if ag == adv:
                            actor_acts.append(logits)
                            adv_logits = logits  # for regularization
                        else:
                            actor_acts.append(logits.detach())
                    actor_input = torch.cat(actor_acts, dim=-1)
                    actor_loss = -agents[adv].critic(s_states, actor_input).mean()
                    regularization = (adv_logits ** 2).mean()
                    agents[adv].update_actor(actor_loss + 1e-3 * regularization)
    env.close()


if __name__ == "__main__":
    train(40, 50, 500, 10, 0.95, 0.001)
