"""
Evaluation script for trained MADDPG models.
Loads saved models and generates GIFs of agent behavior.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

from maddpg import MADDPG
from metrics import create_metrics_tracker, get_all_metric_names


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


def make_env(env_name, max_cycles=25, render_mode='rgb_array'):
    """Create environment for evaluation."""
    env_map = {
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

    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(env_map.keys())}")

    env = env_map[env_name].parallel_env(max_cycles=max_cycles, render_mode=render_mode)
    env.reset()

    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in env.agents}
    action_dims = {agent: env.action_space(agent).n for agent in env.agents}

    return env, obs_dims, action_dims


def evaluate(args):
    """Run evaluation episodes and generate GIFs."""
    # Verify model directory exists
    model_dir = os.path.join('./results', args.env_name, args.run_id)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Setup GIF directory
    gif_dir = os.path.join(model_dir, 'gif')
    os.makedirs(gif_dir, exist_ok=True)
    existing_gifs = len([f for f in os.listdir(gif_dir) if f.endswith('.gif')])

    # Create environment
    env, obs_dims, action_dims = make_env(
        args.env_name, args.episode_length, render_mode='rgb_array'
    )
    agent_ids = list(obs_dims.keys())

    # Initialize and load MADDPG
    maddpg = MADDPG(
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        action_dims=action_dims,
        buffer_capacity=1,  # Not used for evaluation
        device=args.device
    )
    maddpg.load(model_dir)
    print(f"Loaded model from: {model_dir}")

    # Identify agent team for scoring
    agent_team = get_agent_team(args.env_name, agent_ids)
    adversary_team = [a for a in agent_ids if a not in agent_team]
    print(f"Agent team: {agent_team}")

    # Track rewards
    episode_rewards = {agent: np.zeros(args.num_episodes) for agent in agent_ids}
    agent_scores = np.zeros(args.num_episodes)

    # Environment-specific metrics
    metrics_tracker = create_metrics_tracker(args.env_name, agent_ids)
    metric_names = get_all_metric_names(args.env_name)
    episode_metrics = {name: [] for name in metric_names}

    # Run evaluation episodes
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agent_ids}
        metrics_tracker.reset_episode()
        frames = []

        while env.agents:
            # Select actions (deterministic for evaluation)
            actions = maddpg.select_actions(obs, explore=False)

            # Step environment
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in terminations.keys()}

            # Update metrics
            metrics_tracker.update(obs, actions, rewards, next_obs, dones, env)

            # Capture frame
            frames.append(Image.fromarray(env.render()))

            # Accumulate rewards
            for agent in env.agents:
                episode_reward[agent] += rewards[agent]

            obs = next_obs

        # Record episode rewards
        for agent in agent_ids:
            episode_rewards[agent][episode] = episode_reward[agent]
        agent_scores[episode] = sum(episode_reward[a] for a in agent_team)

        # Record metrics
        ep_metrics = metrics_tracker.get_episode_metrics()
        for name, value in ep_metrics.items():
            if name in episode_metrics:
                episode_metrics[name].append(value)

        # Save GIF
        gif_path = os.path.join(gif_dir, f'episode_{existing_gifs + episode + 1}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # ms per frame
            loop=0
        )

        # Print episode summary
        print(f"Episode {episode + 1}: Agent Score = {agent_scores[episode]:.1f} | Saved: {gif_path}")

    env.close()

    # Print evaluation summary
    mean_agent_score = np.mean(agent_scores)
    std_agent_score = np.std(agent_scores)
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {args.env_name}")
    print(f"{'='*60}")
    print(f"Agent Score: {mean_agent_score:.2f} +/- {std_agent_score:.2f}")
    print(f"Min: {np.min(agent_scores):.2f}, Max: {np.max(agent_scores):.2f}")

    # Print task-specific metrics
    if episode_metrics:
        print(f"\nTask-Specific Metrics:")
        print(f"-" * 40)
        for name, values in episode_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {name}: {mean_val:.2f} +/- {std_val:.2f}")
    print(f"{'='*60}")

    # Plot evaluation results
    if args.num_episodes > 1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        x = range(1, args.num_episodes + 1)

        # Top: Agent Score
        ax1 = axes[0]
        ax1.bar(x, agent_scores, color='blue', alpha=0.7)
        ax1.axhline(y=mean_agent_score, color='red', linestyle='--', label=f'Mean: {mean_agent_score:.2f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Agent Score')
        ax1.set_title(f'MADDPG Evaluation: {args.env_name} - Agent Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Per-agent rewards
        ax2 = axes[1]
        for agent in agent_ids:
            ax2.plot(x, episode_rewards[agent], marker='o', label=agent)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Per-Agent Rewards')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(model_dir, 'evaluation_results.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nEvaluation plot saved to: {plot_path}")

    print(f"GIFs saved to: {gif_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MADDPG model')

    parser.add_argument('env_name', type=str,
                        choices=[
                            'simple_v3',
                            'simple_adversary_v3',
                            'simple_crypto_v3',
                            'simple_push_v3',
                            'simple_reference_v3',
                            'simple_speaker_listener_v4',
                            'simple_spread_v3',
                            'simple_tag_v3',
                            'simple_world_comm_v3',
                        ],
                        help='Environment name')
    parser.add_argument('run_id', type=str,
                        help='Run ID (folder name in results/env_name/)')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--episode_length', type=int, default=50,
                        help='Maximum steps per episode')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
