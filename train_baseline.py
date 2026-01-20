"""
Training script for baseline RL algorithms using Stable-Baselines3.
Implements Independent Learners approach with parameter sharing.

Supported algorithms: DQN, PPO, A2C, TRPO
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import supersuit as ss
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
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import TRPO

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

# Agent team definitions (same as train.py)
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


def make_sb3_env(env_name, max_cycles=25):
    """Create SB3-compatible vectorized environment."""
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}")

    env = ENV_MAP[env_name].parallel_env(max_cycles=max_cycles)
    # Pad observations to make them consistent across agents
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    # Convert to vectorized env (each agent becomes a separate env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, base_class="stable_baselines3")
    return env


def make_eval_env(env_name, max_cycles=25):
    """Create evaluation environment (native PettingZoo parallel)."""
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}")
    return ENV_MAP[env_name].parallel_env(max_cycles=max_cycles)


class RewardCallback(BaseCallback):
    """Callback to track episode rewards during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = None

    def _on_training_start(self):
        self.current_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        # Accumulate rewards
        self.current_rewards += self.locals.get('rewards', np.zeros_like(self.current_rewards))

        # Check for episode ends
        dones = self.locals.get('dones', np.zeros(self.training_env.num_envs, dtype=bool))
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0

        return True


def pad_observation(obs, target_shape):
    """Pad observation to target shape with zeros."""
    if obs.shape[0] >= target_shape:
        return obs[:target_shape]
    padded = np.zeros(target_shape, dtype=obs.dtype)
    padded[:obs.shape[0]] = obs
    return padded


def evaluate_model(model, env_name, num_episodes=1000, max_cycles=25):
    """Evaluate trained model and collect metrics."""
    env = make_eval_env(env_name, max_cycles)
    env.reset()

    agent_ids = env.agents.copy()
    agent_team = get_agent_team(env_name, agent_ids)
    adversary_team = [a for a in agent_ids if a not in agent_team]

    # Get model's expected observation shape (for padding)
    expected_obs_shape = model.observation_space.shape[0]

    # Initialize metrics
    metrics_tracker = create_metrics_tracker(env_name, agent_ids)
    metric_names = get_all_metric_names(env_name)

    episode_rewards = {agent: np.zeros(num_episodes) for agent in agent_ids}
    agent_scores = np.zeros(num_episodes)
    adversary_scores = np.zeros(num_episodes)
    episode_metrics = {name: np.zeros(num_episodes) for name in metric_names}

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agent_ids}
        metrics_tracker.reset_episode()

        while env.agents:
            # Get actions from model for each agent
            actions = {}
            for agent in env.agents:
                agent_obs = pad_observation(obs[agent], expected_obs_shape)
                action, _ = model.predict(agent_obs, deterministic=True)
                # Clip action to valid range for this agent's action space
                action_n = env.action_space(agent).n
                actions[agent] = int(action) % action_n

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in terminations.keys()}

            # Update metrics
            metrics_tracker.update(obs, actions, rewards, next_obs, dones, env)

            # Accumulate rewards
            for agent in rewards.keys():
                episode_reward[agent] += rewards[agent]

            obs = next_obs

        # Record episode data
        for agent in agent_ids:
            episode_rewards[agent][episode] = episode_reward.get(agent, 0)

        agent_scores[episode] = sum(episode_reward.get(a, 0) for a in agent_team)
        adversary_scores[episode] = sum(episode_reward.get(a, 0) for a in adversary_team)

        # Record metrics
        ep_metrics = metrics_tracker.get_episode_metrics()
        for name, value in ep_metrics.items():
            if name in episode_metrics:
                episode_metrics[name][episode] = value

        if (episode + 1) % 100 == 0:
            print(f"  Evaluation: {episode + 1}/{num_episodes} episodes")

    env.close()

    return {
        'per_agent': episode_rewards,
        'agent_score': agent_scores,
        'adversary_score': adversary_scores,
        'agent_team': agent_team,
        'adversary_team': adversary_team,
        'metrics': episode_metrics,
    }


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
    result_dir = os.path.join('./results', args.algorithm, args.env_name)
    os.makedirs(result_dir, exist_ok=True)
    run_num = len([d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]) + 1
    result_dir = os.path.join(result_dir, str(run_num))
    os.makedirs(result_dir)

    # Save training arguments
    args_dict = vars(args).copy()
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    print(f"Algorithm: {args.algorithm}")
    print(f"Environment: {args.env_name}")
    print(f"Results will be saved to: {result_dir}")

    # Create training environment
    env = make_sb3_env(args.env_name, args.episode_length)

    # Create model based on algorithm
    if args.algorithm == 'dqn':
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1,
        )
    elif args.algorithm == 'ppo':
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
        )
    elif args.algorithm == 'a2c':
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=5,
            gamma=args.gamma,
            verbose=1,
        )
    elif args.algorithm == 'trpo':
        model = TRPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Train with callback
    callback = RewardCallback()
    print(f"\nTraining for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Save model
    model.save(os.path.join(result_dir, 'model'))
    print(f"Model saved to: {result_dir}/model")

    # Evaluate model
    print(f"\nEvaluating for {args.eval_episodes} episodes...")
    rewards_data = evaluate_model(model, args.env_name, args.eval_episodes, args.episode_length)

    # Save rewards data
    with open(os.path.join(result_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(rewards_data, f)

    # Print summary
    agent_scores = rewards_data['agent_score']
    print(f"\n{'='*60}")
    print(f"Training Summary: {args.algorithm} on {args.env_name}")
    print(f"{'='*60}")
    print(f"Agent Score: {np.mean(agent_scores):.2f} +/- {np.std(agent_scores):.2f}")
    if rewards_data['adversary_team']:
        adv_scores = rewards_data['adversary_score']
        print(f"Adversary Score: {np.mean(adv_scores):.2f} +/- {np.std(adv_scores):.2f}")
    print(f"{'='*60}")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, len(agent_scores) + 1)
    ax.plot(x, agent_scores, alpha=0.3, color='blue', label='Agent Score (raw)')
    ax.plot(x, compute_running_reward(agent_scores), color='blue', linewidth=2, label='Agent Score (smoothed)')
    if rewards_data['adversary_team']:
        adv_scores = rewards_data['adversary_score']
        ax.plot(x, adv_scores, alpha=0.3, color='red', label='Adversary Score (raw)')
        ax.plot(x, compute_running_reward(adv_scores), color='red', linewidth=2, label='Adversary Score (smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title(f'{args.algorithm.upper()} Evaluation: {args.env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to: {result_dir}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train baseline RL algorithms')

    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['dqn', 'ppo', 'a2c', 'trpo'],
                        help='Algorithm to train')
    parser.add_argument('--env_name', type=str, required=True,
                        choices=list(ENV_MAP.keys()),
                        help='Environment name')
    parser.add_argument('--timesteps', type=int, default=750000,
                        help='Total training timesteps')
    parser.add_argument('--episode_length', type=int, default=25,
                        help='Maximum steps per episode')
    parser.add_argument('--eval_episodes', type=int, default=1000,
                        help='Number of evaluation episodes')

    # Algorithm hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='Replay buffer size (for DQN)')
    parser.add_argument('--learning_starts', type=int, default=10000,
                        help='Steps before learning starts (for DQN)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
