"""
Evaluation script for trained MADDPG models.
Loads saved models and generates GIFs of agent behavior.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3, simple_spread_v3

from maddpg import MADDPG


def make_env(env_name, max_cycles=25, render_mode='rgb_array'):
    """Create environment for evaluation."""
    env_map = {
        'simple_tag_v3': simple_tag_v3,
        'simple_adversary_v3': simple_adversary_v3,
        'simple_spread_v3': simple_spread_v3,
    }

    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}")

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

    # Track rewards
    episode_rewards = {agent: np.zeros(args.num_episodes) for agent in agent_ids}

    # Run evaluation episodes
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in agent_ids}
        frames = []

        while env.agents:
            # Select actions (deterministic for evaluation)
            actions = maddpg.select_actions(obs, explore=False)

            # Step environment
            next_obs, rewards, _, _, _ = env.step(actions)

            # Capture frame
            frames.append(Image.fromarray(env.render()))

            # Accumulate rewards
            for agent in env.agents:
                episode_reward[agent] += rewards[agent]

            obs = next_obs

        # Record episode rewards
        for agent in agent_ids:
            episode_rewards[agent][episode] = episode_reward[agent]

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
        total = sum(episode_reward.values())
        print(f"Episode {episode + 1}: Total reward = {total:.1f} | Saved: {gif_path}")

    env.close()

    # Plot evaluation results
    if args.num_episodes > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(1, args.num_episodes + 1)

        for agent in agent_ids:
            ax.plot(x, episode_rewards[agent], marker='o', label=agent)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'MADDPG Evaluation: {args.env_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(model_dir, 'evaluation_results.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nEvaluation plot saved to: {plot_path}")

    print(f"\nGIFs saved to: {gif_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MADDPG model')

    parser.add_argument('env_name', type=str,
                        choices=['simple_tag_v3', 'simple_adversary_v3', 'simple_spread_v3'],
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
