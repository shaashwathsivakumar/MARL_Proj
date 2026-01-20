"""
Training script for vanilla REINFORCE algorithm.
Implements Independent Learners with parameter sharing approach.
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pettingzoo.mpe import (
    simple_v3,
    simple_adversary_v3,
    simple_crypto_v3,
    simple_push_v3,
    simple_reference_v3,
    simple_speaker_listener_v4,
    simple_spread_v3,
    simple_tag_v3,
    simple_world_comm_v3,
)

from metrics import create_metrics_tracker, get_all_metric_names


# Environment mapping
ENV_MAP = {
    'simple_v3': simple_v3,
    'simple_adversary_v3': simple_adversary_v3,
    'simple_crypto_v3': simple_crypto_v3,
    'simple_push_v3': simple_push_v3,
    'simple_reference_v3': simple_reference_v3,
    'simple_speaker_listener_v4': simple_speaker_listener_v4,
    'simple_spread_v3': simple_spread_v3,
    'simple_tag_v3': simple_tag_v3,
    'simple_world_comm_v3': simple_world_comm_v3,
}

# Agent team definitions
AGENT_TEAMS = {
    'simple_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_reference_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_speaker_listener_v4': lambda agents: agents,
    'simple_spread_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_adversary_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_crypto_v3': lambda agents: [a for a in agents if a.startswith('alice') or a.startswith('bob')],
    'simple_push_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_tag_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_world_comm_v3': lambda agents: [a for a in agents if a.startswith('agent')],
}


def get_agent_team(env_name, agent_ids):
    """Get list of agents in the 'agent' team for score computation."""
    if env_name in AGENT_TEAMS:
        return AGENT_TEAMS[env_name](agent_ids)
    return [a for a in agent_ids if a.startswith('agent')]


class PolicyNetwork(nn.Module):
    """Simple MLP policy network for REINFORCE."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_action(self, obs, deterministic=False):
        """Get action from observation."""
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = None
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob


class REINFORCE:
    """Vanilla REINFORCE algorithm with parameter sharing."""

    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.95, device='cpu'):
        self.device = device
        self.gamma = gamma

        self.policy = PolicyNetwork(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Episode storage
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs, deterministic=False):
        """Select action given observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob = self.policy.get_action(obs_tensor, deterministic)

        if not deterministic:
            self.log_probs.append(log_prob)

        return action.item()

    def store_reward(self, reward):
        """Store reward for current step."""
        self.rewards.append(reward)

    def update(self):
        """Update policy using collected episode data."""
        if len(self.rewards) == 0:
            return 0.0

        # Compute returns (reward-to-go)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient loss
        log_probs = torch.cat(self.log_probs)
        loss = -(log_probs * returns).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode storage
        self.log_probs = []
        self.rewards = []

        return loss.item()

    def save(self, path):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def make_env(env_name, max_cycles=25):
    """Create PettingZoo parallel environment."""
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}")
    env = ENV_MAP[env_name].parallel_env(max_cycles=max_cycles)
    env.reset()
    return env


def pad_observation(obs, target_shape):
    """Pad observation to target shape with zeros."""
    if obs.shape[0] >= target_shape:
        return obs[:target_shape]
    padded = np.zeros(target_shape, dtype=obs.dtype)
    padded[:obs.shape[0]] = obs
    return padded


def compute_running_reward(rewards, window=100):
    """Compute running average of rewards."""
    running = np.zeros_like(rewards)
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        running[i] = np.mean(rewards[start:i + 1])
    return running


def train(args):
    """Main training function."""
    # Setup results directory
    result_dir = os.path.join('./results', 'reinforce', args.env_name)
    os.makedirs(result_dir, exist_ok=True)
    run_num = len([d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]) + 1
    result_dir = os.path.join(result_dir, str(run_num))
    os.makedirs(result_dir)

    # Save training arguments
    args_dict = vars(args).copy()
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    print(f"Algorithm: REINFORCE")
    print(f"Environment: {args.env_name}")
    print(f"Results will be saved to: {result_dir}")

    # Create environment
    env = make_env(args.env_name, args.episode_length)
    agent_ids = list(env.agents)

    # Get max dimensions across all agents for parameter sharing
    obs_dim = max(env.observation_space(a).shape[0] for a in agent_ids)
    action_dim = max(env.action_space(a).n for a in agent_ids)

    # Identify teams
    agent_team = get_agent_team(args.env_name, agent_ids)
    adversary_team = [a for a in agent_ids if a not in agent_team]

    print(f"Agents: {agent_ids}")
    print(f"Agent team: {agent_team}")
    print(f"Max Obs dim: {obs_dim}, Max Action dim: {action_dim}")

    # Create REINFORCE agent (shared policy for all agents)
    device = torch.device(args.device)
    reinforce = REINFORCE(obs_dim, action_dim, lr=args.lr, gamma=args.gamma, device=device)

    # Metrics tracking
    metrics_tracker = create_metrics_tracker(args.env_name, agent_ids)
    metric_names = get_all_metric_names(args.env_name)

    # Training storage
    episode_rewards = {agent: np.zeros(args.num_episodes) for agent in agent_ids}
    agent_scores = np.zeros(args.num_episodes)
    adversary_scores = np.zeros(args.num_episodes)
    episode_metrics = {name: np.zeros(args.num_episodes) for name in metric_names}

    # Training loop
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agent_ids}
        metrics_tracker.reset_episode()

        # Storage for this episode (per-agent)
        episode_log_probs = {agent: [] for agent in agent_ids}
        episode_agent_rewards = {agent: [] for agent in agent_ids}

        while env.agents:
            actions = {}
            for agent in env.agents:
                # Pad observation and select action
                padded_obs = pad_observation(obs[agent], obs_dim)
                obs_tensor = torch.FloatTensor(padded_obs).unsqueeze(0).to(device)
                action, log_prob = reinforce.policy.get_action(obs_tensor, deterministic=False)
                # Clip action to valid range for this agent's action space
                agent_action_n = env.action_space(agent).n
                actions[agent] = action.item() % agent_action_n
                episode_log_probs[agent].append(log_prob)

            # Step environment
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in terminations.keys()}

            # Store rewards
            for agent in rewards.keys():
                episode_agent_rewards[agent].append(rewards[agent])
                episode_reward[agent] += rewards[agent]

            # Update metrics
            metrics_tracker.update(obs, actions, rewards, next_obs, dones, env)

            obs = next_obs

        # Aggregate all agents' trajectories for a single update (avoids gradient issues)
        all_log_probs = []
        all_rewards = []
        for agent in agent_ids:
            if len(episode_agent_rewards[agent]) > 0:
                all_log_probs.extend(episode_log_probs[agent])
                all_rewards.extend(episode_agent_rewards[agent])

        # Update policy with combined trajectory
        reinforce.log_probs = all_log_probs
        reinforce.rewards = all_rewards
        total_loss = reinforce.update()

        # Record episode data
        for agent in agent_ids:
            episode_rewards[agent][episode] = episode_reward[agent]

        agent_scores[episode] = sum(episode_reward[a] for a in agent_team)
        adversary_scores[episode] = sum(episode_reward[a] for a in adversary_team)

        # Record metrics
        ep_metrics = metrics_tracker.get_episode_metrics()
        for name, value in ep_metrics.items():
            if name in episode_metrics:
                episode_metrics[name][episode] = value

        # Logging
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode + 1}/{args.num_episodes} | "
                  f"AgentScore: {agent_scores[episode]:.1f} | "
                  f"Loss: {total_loss:.4f}")

    env.close()

    # Save model
    reinforce.save(os.path.join(result_dir, 'model.pt'))

    # Save rewards data
    rewards_data = {
        'per_agent': episode_rewards,
        'agent_score': agent_scores,
        'adversary_score': adversary_scores,
        'agent_team': agent_team,
        'adversary_team': adversary_team,
        'metrics': episode_metrics,
    }
    with open(os.path.join(result_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(rewards_data, f)

    # Print summary
    last_n = min(1000, args.num_episodes)
    print(f"\n{'='*60}")
    print(f"Training Summary: REINFORCE on {args.env_name}")
    print(f"{'='*60}")
    print(f"Agent Score (last {last_n}): {np.mean(agent_scores[-last_n:]):.2f} +/- {np.std(agent_scores[-last_n:]):.2f}")
    if adversary_team:
        print(f"Adversary Score (last {last_n}): {np.mean(adversary_scores[-last_n:]):.2f} +/- {np.std(adversary_scores[-last_n:]):.2f}")
    print(f"{'='*60}")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, args.num_episodes + 1)
    ax.plot(x, agent_scores, alpha=0.3, color='blue', label='Agent Score (raw)')
    ax.plot(x, compute_running_reward(agent_scores), color='blue', linewidth=2, label='Agent Score (smoothed)')
    if adversary_team:
        ax.plot(x, adversary_scores, alpha=0.3, color='red', label='Adversary Score (raw)')
        ax.plot(x, compute_running_reward(adversary_scores), color='red', linewidth=2, label='Adversary Score (smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title(f'REINFORCE Training: {args.env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to: {result_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train REINFORCE algorithm')

    parser.add_argument('--env_name', type=str, required=True,
                        choices=list(ENV_MAP.keys()),
                        help='Environment name')
    parser.add_argument('--num_episodes', type=int, default=30000,
                        help='Total training episodes')
    parser.add_argument('--episode_length', type=int, default=25,
                        help='Maximum steps per episode')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Episodes between log messages')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
