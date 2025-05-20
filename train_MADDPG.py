import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3


class Actor(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, obs: torch.Tensor):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_size, actions_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + actions_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        x = torch.cat((state, actions), dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))  # Apply ReLU, not in-place
        value = self.fc2(x)
        return value


class MADDPGAgent:
    def __init__(self, obs_size, action_size, state_size, joint_action_size, lr=0.001, device="cpu"):
        self.actor = Actor(obs_size, action_size).to(device)
        self.critic = Critic(state_size, joint_action_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def get_action(self, obs, model_out=False, device="cpu"):
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
    
    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def update_targets(self, tau):
        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)


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


def save(self, reward):
    torch.save(
        {name: agent.actor.state_dict() for name, agent in self.agents.items()},
        os.path.join(self.res_dir, 'model.pt')
    )
    with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump({'rewards': reward}, f)


def plot(reward):
    plt.plot(reward)
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.grid(True)
    plt.show()


def train(max_episode_count, max_episode_length, max_capacity, minibatch_size, gamma, lr, noise_param, device="cpu"):
    env = simple_tag_v3.parallel_env(render_mode=None, max_cycles=max_episode_length)
    observations, infos = env.reset()
    all_agents = env.agents
    agents = {agent: MADDPGAgent(len(observations[agent]), env.action_space(agent).n, len(env.state()), sum(env.action_space(a).n for a in all_agents), lr, device) for agent in all_agents}
    D = ReplayBuffer(max_capacity)
    episode_rewards = []
    for i in range(max_episode_count):
        observations, infos = env.reset()
        total_reward = 0
        while env.agents:
            state = env.state()
            actions = {}
            logits_store = {}
            for adv in all_agents:
                logits = agents[adv].get_action(observations[adv], model_out=True, device=device)[1]
                noise = torch.randn_like(logits) * noise_param
                action = torch.argmax(logits + noise).item()
                actions[adv] = action
                logits_store[adv] = logits
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())
            next_state = env.state()
            D.push(state, actions, rewards, next_state, observations)
            observations = next_obs
            if len(D.buffer) >= minibatch_size:
                for adv in all_agents:
                    samples = D.sample(minibatch_size)
                    s_states = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in samples]).to(device)
                    s_next_states = torch.stack([torch.tensor(s[3], dtype=torch.float32) for s in samples]).to(device)
                    s_rewards = torch.tensor([s[2][adv] for s in samples], dtype=torch.float32).to(device)
                    next_acts = []
                    for ag in all_agents:
                        obs_batch = torch.stack([torch.tensor(s[4][ag], dtype=torch.float32) for s in samples]).to(
                            device)
                        logits = agents[ag].target_actor(obs_batch)
                        next_action_indices = torch.argmax(logits, dim = -1)
                        next_actions_onehot = F.one_hot(next_action_indices, num_classes=env.action_space(ag).n).float()
                        next_acts.append(next_actions_onehot)
                    next_joint_actions = torch.cat(next_acts, dim=-1)
                    with torch.no_grad():
                        target_q = s_rewards + gamma * agents[adv].target_critic(s_next_states, next_joint_actions).squeeze(1)
                    joint_acts = []
                    for ag in all_agents:
                        actions = torch.tensor([s[1][ag] for s in samples], dtype=torch.int64)
                        joint_acts.append(F.one_hot(actions, num_classes=env.action_space(ag).n).float().to(device))
                    current_q = agents[adv].critic(s_states, torch.cat(joint_acts, dim=-1)).squeeze(1)
                    critic_loss = nn.MSELoss()(current_q, target_q)
                    agents[adv].update_critic(critic_loss)
                    # actor update
                    actor_acts = []
                    for ag in all_agents:
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
                for adv in all_agents:
                    agents[adv].update_targets(tau = 0.01)
        episode_rewards.append(total_reward)        
    env.close()
    #plot(episode_rewards)
    #save(episode_rewards)


if __name__ == "__main__":
    train(40, 50, 50, 10, 0.95, 0.001, 0.3, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
